# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings from ONNX ops (baseline opset window 17-22) to Core AI dialect ops."""

from collections.abc import Callable
from typing import Any

from . import (
    _activations,
    _attention,
    _controlflow,
    _conv,
    _elementwise,
    _indexing,
    _matmul,
    _norm,
    _quantization,
    _recurrent,
    _reduce,
    _shape,
)

_REGISTRIES = (
    _elementwise.REGISTRY,
    _shape.REGISTRY,
    _reduce.REGISTRY,
    _indexing.REGISTRY,
    _matmul.REGISTRY,
    _controlflow.REGISTRY,
    _conv.REGISTRY,
    _norm.REGISTRY,
    _quantization.REGISTRY,
    _recurrent.REGISTRY,
    _attention.REGISTRY,
    _activations.REGISTRY,
)

_onnx_to_core_resolver: dict[str, Callable[..., Any]] = {}
for _registry in _REGISTRIES:
    for _key, _fn in _registry.items():
        if _key in _onnx_to_core_resolver:
            raise RuntimeError(f"duplicate ONNX op lowering registered: '{_key}'")
        _onnx_to_core_resolver[_key] = _fn

__all__ = ["_onnx_to_core_resolver"]
