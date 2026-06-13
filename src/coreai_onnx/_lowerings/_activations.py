# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for fused activation nodes produced by ``fuse_activations``.

ONNX has no SiLU op (through opset 22), so exporters decompose it into
Sigmoid -> Mul; ``coreai_onnx._fusion.fuse_activations`` rewrites that
pair into a custom ``coreai_onnx::Silu`` node, lowered here to the runtime's
fused ``coreai.silu`` kernel.  (Fused GELU chains become standard ONNX Gelu
nodes and reuse the existing lowering in ``_elementwise``.)
"""

from collections.abc import Callable
from typing import Any

import onnx

from .._ir import Location, Value
from .._ir import coreai_dialect as coreai
from .._utils import operands


def replace_silu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return coreai.silu(x)


REGISTRY: dict[str, Callable[..., Any]] = {
    "coreai_onnx::Silu": replace_silu,
}
