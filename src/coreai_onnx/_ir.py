# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Single point of contact with Core AI's private compiler APIs.

Everything under ``coreai._compiler`` is not part of coreai-core's public,
semver-stable surface (coreai-torch consumes the same modules the same way).
Importing those names from here — instead of from ``coreai._compiler.*``
directly — keeps the breakage surface of a coreai-core upgrade to this one
module plus :func:`program_from_module` below.
"""

from typing import cast

from coreai._compiler.dialects import coreai as coreai_dialect
from coreai._compiler.ir import (
    ArrayAttr,
    Attribute,
    BF16Type,
    BoolAttr,
    Context,
    DenseElementsAttr,
    DictAttr,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    FloatAttr,
    FloatType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    OpResult,
    OpResultList,
    RankedTensorType,
    ShapedType,
    StringAttr,
    Type,
    Value,
)
from coreai.authoring import AIProgram


def tensor_type(v: Value) -> RankedTensorType:
    """Typed view of a Value's type.

    Every value in a ``coreai.graph`` is a ranked tensor, but the upstream
    binding types ``Value.type`` as the MLIR base ``Type``; this cast is the
    blessed way to reach ``shape``/``rank``/``element_type`` without tripping
    the type checker. Zero runtime cost.
    """
    return cast(RankedTensorType, v.type)


def program_from_module(module: Module) -> AIProgram:
    """Wrap a finished MLIR module as an AIProgram.

    ``AIProgram._from_mlir_module`` is private but is the only constructor
    coreai-core offers for externally-built modules (coreai-torch uses the
    same entry point).
    """
    return AIProgram._from_mlir_module(module)


__all__ = [
    "AIProgram",
    "ArrayAttr",
    "Attribute",
    "BF16Type",
    "BoolAttr",
    "Context",
    "DenseElementsAttr",
    "DictAttr",
    "F16Type",
    "F32Type",
    "Float8E4M3FNType",
    "Float8E5M2Type",
    "FloatAttr",
    "FloatType",
    "InsertionPoint",
    "IntegerAttr",
    "IntegerType",
    "Location",
    "Module",
    "OpResult",
    "OpResultList",
    "RankedTensorType",
    "ShapedType",
    "StringAttr",
    "Type",
    "Value",
    "coreai_dialect",
    "program_from_module",
    "tensor_type",
]
