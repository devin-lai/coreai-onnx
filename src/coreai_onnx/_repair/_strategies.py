# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Known-safe ONNX rewrites for Core AI runtime limitations."""

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

    ``applies`` decides whether the strategy runs. ``detect`` returns details
    for the repair record and may be empty even when the rewrite is needed.
    """

    name: str
    summary: str
    detect: Callable[[onnx.ModelProto], list[str]]
    apply: Callable[[onnx.ModelProto], None]
    applies: Callable[[onnx.ModelProto], bool]


def _float16_inputs(model: onnx.ModelProto) -> list[str]:
    return [
        vi.name for vi in model.graph.input if vi.type.tensor_type.elem_type == _F16
    ]


def _graph_has_float16(graph: onnx.GraphProto) -> bool:
    """Return whether the graph contains a float16 declaration we can rewrite."""
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.type.tensor_type.elem_type == _F16:
            return True
    if any(init.data_type == _F16 for init in graph.initializer):
        return True
    for node in graph.node:
        for attr in node.attribute:
            if attr.name in ("to", "dtype") and attr.i == _F16:
                return True
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == _F16:
                return True
    return False


def _has_float16(model: onnx.ModelProto) -> bool:
    """Return whether any graph or nested subgraph uses float16."""
    found = False

    def _check(graph: onnx.GraphProto) -> None:
        nonlocal found
        found = found or _graph_has_float16(graph)

    apply_graphs_topdown(model.graph, _check)
    return found


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
            # Cast(to=FLOAT16), RandomNormal(dtype=FLOAT16), etc.
            if attr.name in ("to", "dtype") and attr.i == _F16:
                attr.i = _F32
            # Constant / ConstantOfShape value tensors
            if attr.type == onnx.AttributeProto.TENSOR:
                _promote_float16_tensor(attr.t)


def _promote_float16(model: onnx.ModelProto) -> None:
    """Rewrite float16 declarations to float32 throughout the model."""
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
    applies=_has_float16,
)

STRATEGIES: tuple[RepairStrategy, ...] = (PROMOTE_FLOAT16,)
