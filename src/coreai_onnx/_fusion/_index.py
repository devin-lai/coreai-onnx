# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tensor/scalar lookup tables shared by the fusion passes' matchers."""

import math
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import numpy_helper

from .._utils import iter_subgraph_inputs

_FLOAT_ELEM_TYPES = (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16)

# The composite declaration stores scale as float32; a scale that does not
# survive the round trip (inf/NaN, or 1/denormal overflowing f32) would
# corrupt the fused kernel output.
_F32_MAX = float(np.finfo(np.float32).max)

# A dimension is an int (static), a str (symbolic dim_param), or None (unknown).
_Dims = tuple[int | str | None, ...]


@dataclass
class _TensorInfo:
    elem_type: int
    dims: _Dims


def _graph_tensor_infos(graph: onnx.GraphProto) -> dict[str, _TensorInfo]:
    infos: dict[str, _TensorInfo] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        tt = vi.type.tensor_type
        if not tt.shape.dim:
            continue
        dims = tuple(
            d.dim_value
            if d.HasField("dim_value")
            else (d.dim_param if d.dim_param else None)
            for d in tt.shape.dim
        )
        infos[vi.name] = _TensorInfo(tt.elem_type, dims)
    for init in graph.initializer:
        infos[init.name] = _TensorInfo(init.data_type, tuple(init.dims))
    return infos


def _graph_scalars(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    # Only size-1 initializers can be scale constants; skipping the rest
    # avoids materializing every weight tensor of a large model.
    return {
        i.name: numpy_helper.to_array(i)
        for i in graph.initializer
        if math.prod(i.dims) == 1
    }


@dataclass
class _GraphIndex:
    """Lookup tables over one (sub)graph used by the matchers."""

    by_output: dict[str, onnx.NodeProto]
    consumer_count: dict[str, int]
    graph_outputs: set[str]
    infos: dict[str, _TensorInfo]
    scalars: dict[str, np.ndarray]  # initializers, for scale extraction

    @classmethod
    def build(
        cls,
        graph: onnx.GraphProto,
        infos: dict[str, _TensorInfo],
        scalars: dict[str, np.ndarray],
    ) -> "_GraphIndex":
        by_output = {out: n for n in graph.node for out in n.output}
        consumer_count: dict[str, int] = {}
        for n in graph.node:
            for name in n.input:
                consumer_count[name] = consumer_count.get(name, 0) + 1
            # Values captured inside If/Loop bodies count as consumers too.
            for name in iter_subgraph_inputs(n):
                consumer_count[name] = consumer_count.get(name, 0) + 1
        return cls(
            by_output=by_output,
            consumer_count=consumer_count,
            graph_outputs={o.name for o in graph.output},
            infos=infos,
            scalars=scalars,
        )

    def is_internal(self, name: str) -> bool:
        """True if *name* is consumed exactly once and is not a graph output."""
        return name not in self.graph_outputs and self.consumer_count.get(name) == 1

    def scalar(self, name: str) -> float | None:
        arr = self.scalars.get(name)
        if arr is None or arr.size != 1:
            return None
        return float(arr.reshape(()))

    def dims(self, name: str) -> _Dims | None:
        info = self.infos.get(name)
        return info.dims if info is not None else None
