# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Known-safe conversion repairs.

Each strategy is a documented, semantics-preserving ONNX→ONNX rewrite that
removes a Core AI runtime limitation. Strategies never change results beyond
the precision of the original dtype; the engine re-verifies parity against
ONNX Runtime before accepting a repair, so an incomplete rewrite fails closed
(rejected) rather than producing a wrong model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

from .._utils import apply_graphs_topdown

_F16 = onnx.TensorProto.FLOAT16
_F32 = onnx.TensorProto.FLOAT


@dataclass(frozen=True)
class RepairStrategy:
    """One known-safe repair.

    ``detect`` returns the graph-input names that trigger the strategy (empty
    if it does not apply); ``apply`` performs the in-place rewrite. ``summary``
    documents the fix for the repair record and the user.
    """

    name: str
    summary: str
    detect: Callable[[onnx.ModelProto], list[str]]
    apply: Callable[[onnx.ModelProto], None]


def _float16_inputs(model: onnx.ModelProto) -> list[str]:
    return [
        vi.name for vi in model.graph.input if vi.type.tensor_type.elem_type == _F16
    ]


def _promote_float16_tensor(tensor: onnx.TensorProto) -> None:
    if tensor.data_type == _F16:
        arr = numpy_helper.to_array(tensor).astype(np.float32)
        name = tensor.name
        tensor.CopyFrom(numpy_helper.from_array(arr, name))


def _promote_graph_float16(graph: onnx.GraphProto) -> None:
    """Promote every float16 declaration in *graph* to float32 (one level)."""
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == _F16:
            tt.elem_type = _F32
    for init in graph.initializer:
        _promote_float16_tensor(init)
    for node in graph.node:
        for attr in node.attribute:
            # Cast(to=FLOAT16) / RandomNormal(dtype=FLOAT16) / ... → FLOAT
            if attr.name in ("to", "dtype") and attr.i == _F16:
                attr.i = _F32
            # Constant / ConstantOfShape value tensors
            if attr.type == onnx.AttributeProto.TENSOR:
                _promote_float16_tensor(attr.t)


def _promote_float16(model: onnx.ModelProto) -> None:
    """Rewrite a float16 model to float32 throughout (graphs and subgraphs).

    The Core AI runtime cannot load most float16 programs (inputs, arithmetic,
    conv, matmul, softmax all fail with 'Program load failure 0x10004'). Running
    the computation in float32 is the documented workaround; float32 is a strict
    superset of float16, so results match the original within float16 precision.
    Inputs and outputs become float32. Shapes are re-inferred so downstream
    passes see consistent types.
    """
    apply_graphs_topdown(model.graph, _promote_graph_float16)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return
    model.CopyFrom(inferred)


PROMOTE_FLOAT16 = RepairStrategy(
    name="promote_float16_to_float32",
    summary=(
        "promote float16 tensors to float32 (the Core AI runtime cannot load "
        "most float16 programs); inputs and outputs become float32"
    ),
    detect=_float16_inputs,
    apply=_promote_float16,
)

STRATEGIES: tuple[RepairStrategy, ...] = (PROMOTE_FLOAT16,)
