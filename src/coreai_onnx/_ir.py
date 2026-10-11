# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Single point of contact with Core AI's private compiler APIs.

Everything under ``coreai._compiler`` is not part of coreai-core's public,
semver-stable surface (coreai-torch consumes the same modules the same way).
Importing those names from here — instead of from ``coreai._compiler.*``
directly — keeps the breakage surface of a coreai-core upgrade to this one
module: the re-exports plus the :func:`program_from_module`,
:func:`optimize_program` and :func:`mlir_module_of` shims below.
"""

import importlib
from collections.abc import Callable, Iterable
from typing import Any, cast

# Must precede the dialect import: on coreai-core 1.0.0b3 the dialect package
# imports coreai.authoring, which imports the dialect back, so entering that
# cycle from the dialect side raises ImportError.
import coreai.authoring
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
    Operation,
    OpResult,
    OpResultList,
    RankedTensorType,
    ShapedType,
    StringAttr,
    Type,
    Value,
)

AIProgram = coreai.authoring.AIProgram


# Native bindings with no importable type information; resolved by name so a
# coreai-core release that drops or moves one degrades gracefully.
_BINDINGS = "coreai._compiler._mlir_libs._coreaiIR._bindings"


def _native_binding(module: str, name: str) -> Callable[..., Any] | None:
    try:
        binding = getattr(importlib.import_module(f"{_BINDINGS}.{module}"), name)
    except (ImportError, AttributeError):
        return None
    return cast("Callable[..., Any]", binding)


def tensor_type(v: Value) -> RankedTensorType:
    """Typed view of a Value's type.

    Every value in a ``coreai.graph`` is a ranked tensor, but the upstream
    binding types ``Value.type`` as the MLIR base ``Type``; this cast is the
    blessed way to reach ``shape``/``rank``/``element_type`` without tripping
    the type checker. Zero runtime cost.
    """
    return cast(RankedTensorType, v.type)


# A tensor is only materialized as a constant from smaller operands up to this
# many elements; past it the expansion is left to run at inference. Causal
# masks are the motivating case: a 4096-context mask is ~16.8M elements baked
# into the asset when built eagerly, against two 4096-element iotas when not.
EXPANSION_LIMIT = 1 << 16


def _static_element_count(value: Value) -> int:
    shaped = tensor_type(value)
    if not shaped.has_static_shape:
        raise ValueError("dynamic shape")
    count = 1
    for dim in shaped.shape:
        count *= dim
    return count


def operand_values(op: Operation) -> list[Value]:
    """An operation's operands, typed.

    The upstream stubs omit the Sequence protocol on the operand and result
    lists; the runtime bindings are iterable.
    """
    return list(cast("Iterable[Value]", op.operands))


def result_values(op: Operation) -> list[Value]:
    """An operation's results, typed (see :func:`operand_values`)."""
    return list(cast("Iterable[Value]", op.results))


def _should_fold(op: Operation) -> bool:
    """Constant-folding veto: refuse folds that expand past EXPANSION_LIMIT."""
    try:
        produced = sum(_static_element_count(r) for r in result_values(op))
        if produced <= EXPANSION_LIMIT:
            return True
        return produced <= sum(_static_element_count(o) for o in operand_values(op))
    except Exception:
        # Called from the native rewrite driver: never raise into it, and
        # never block a fold this policy cannot reason about.
        return True


def install_fold_guard(context: Context) -> None:
    """Keep the pre-compilation rewrite from densifying large expansions.

    Without it the rewrite folds e.g. Trilu's iota comparison back into a
    dense ``rows x cols`` constant, undoing the lowering's compact form in
    every optimized (i.e. shipped) program.
    """
    register = _native_binding(
        "mlir.dialects.coreai", "register_should_const_folding_hook"
    )
    if register is None:
        return  # this coreai-core has no hook; folding stays unrestricted
    register(callable=_should_fold, context=context)


def program_from_module(module: Module, *, optimize: bool = True) -> AIProgram:
    """Wrap a finished MLIR module as an AIProgram.

    ``AIProgram._from_mlir_module`` is private but is the only constructor
    coreai-core offers for externally-built modules. With ``optimize`` (the
    default) the frontend-agnostic pre-compilation rewrite is applied first,
    so the returned program is ready to save — matching coreai-torch, whose
    ``to_coreai()`` returns an optimized program.
    """
    program = AIProgram._from_mlir_module(module)
    if optimize:
        optimize_program(program)
    return program


def mlir_module_of(program: AIProgram) -> Module:
    """The raw MLIR module behind an AIProgram, across coreai-core layouts.

    coreai-core 1.0.0b3 moved it from ``AIProgram._mlir_module`` to
    ``AIProgram._module._mlir_module`` (behind the authoring ``Module``).
    """
    holder: Any = getattr(program, "_module", program)
    return cast(Module, holder._mlir_module)


def optimize_program(program: AIProgram) -> None:
    """Run Core AI's pre-compilation rewrite on *program*, in place.

    coreai-core 1.0.0b3 removed ``AIProgram.optimize()``: the rewrite now runs
    when an authoring ``Module`` block exits, which never happens for a module
    built through the raw bindings as this converter does, so it is invoked
    directly here. Older releases still expose ``optimize()``, which runs the
    same pipeline.
    """
    legacy_optimize = getattr(program, "optimize", None)
    if legacy_optimize is not None:
        legacy_optimize()
        return
    rewrite = _native_binding("compiler", "run_pre_compilation_rewrite")
    if rewrite is None:
        raise RuntimeError(
            "this coreai-core release exposes neither AIProgram.optimize() nor "
            "the pre-compilation rewrite binding coreai-onnx knows how to "
            "drive; upgrade coreai-onnx, or convert with optimization disabled"
        )
    rewrite(module=mlir_module_of(program))


__all__ = [
    "EXPANSION_LIMIT",
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
    "install_fold_guard",
    "mlir_module_of",
    "operand_values",
    "optimize_program",
    "program_from_module",
    "result_values",
    "tensor_type",
]
