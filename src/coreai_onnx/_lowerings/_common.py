# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared helpers for the ONNX-to-Core-AI op lowerings."""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import (
    BF16Type,
    DenseElementsAttr,
    F16Type,
    FloatType,
    Location,
    OpResult,
    Value,
    tensor_type,
)
from .._ir import coreai_dialect as coreai
from .._utils import operands

_INT32_MAX = 2**31 - 1


def _const_array(v: Value) -> np.ndarray | None:
    """The numpy value of a compile-time coreai.constant, or None if dynamic."""
    if not isinstance(v, OpResult):
        return None  # block argument (graph input)
    op = v.owner
    if op.name != "coreai.constant":
        return None
    try:
        return np.array(DenseElementsAttr(op.attributes["value"]))
    except (ValueError, KeyError):
        return None  # e.g. resource-backed constant or missing attribute


def _require_const(v: Value, op_name: str, what: str) -> np.ndarray:
    """Return the compile-time numpy value of *v*, or raise a descriptive ValueError."""
    arr = _const_array(v)
    if arr is None:
        raise ValueError(f"{op_name}: {what} must be a compile-time constant")
    return arr


# ---------------------------------------------------------------------------
# Simple two-operand factory
# ---------------------------------------------------------------------------


def _binary(coreai_fn: Callable[..., Any]) -> Callable[..., Value]:
    """Return a lowering closure for a straightforward two-input op."""

    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        x, y = operands(values_map, node, [0, 1])
        return coreai_fn(x, y)

    # Strip the "broadcasting_" prefix so the debug name reads replace_add, etc.
    _lower.__name__ = f"replace_{coreai_fn.__name__.removeprefix('broadcasting_')}"
    return _lower


# ---------------------------------------------------------------------------
# Variadic Min / Max  (reduce pairwise over all inputs)
# ---------------------------------------------------------------------------


def _variadic_reduce(
    coreai_fn: Callable[..., Any],
    *,
    propagate_nan: bool = False,
) -> Callable[..., Value]:
    """Reduce all node inputs pairwise with *coreai_fn*.

    propagate_nan: ONNX Min/Max follow np.minimum/maximum — NaN in either
    operand propagates.  The runtime's minimum/maximum return the non-NaN
    operand instead (contradicting the MaximumOp docstring), so for float
    element types the NaNs are patched back in with a where().
    """

    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        # All inputs of the variadic ops are required: a name missing from
        # values_map is an upstream converter bug, so let the KeyError
        # surface instead of silently reducing over a subset.
        vals = [values_map[name] for name in node.input]
        if not vals:
            raise ValueError(f"node '{node.name}': no inputs")
        result = vals[0]
        fixup = propagate_nan and isinstance(
            tensor_type(vals[0]).element_type, FloatType
        )
        for v in vals[1:]:
            merged = coreai_fn(result, v)
            if fixup:
                a, b = _cmp_operand(result), _cmp_operand(v)
                either_nan = coreai.broadcasting_or(
                    coreai.broadcasting_not_equal(a, a),
                    coreai.broadcasting_not_equal(b, b),
                )
                merged = coreai.broadcasting_where(
                    either_nan,
                    coreai.constant(
                        float("nan"), dtype=tensor_type(vals[0]).element_type
                    ),
                    merged,
                )
            result = merged
        return result

    _lower.__name__ = f"replace_{coreai_fn.__name__}_variadic"
    return _lower


# ---------------------------------------------------------------------------
# Unary helpers
# ---------------------------------------------------------------------------


def _unary(coreai_fn: Callable[..., Any]) -> Callable[..., Value]:
    """Return a lowering closure for a one-input op."""

    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        (x,) = operands(values_map, node, [0])
        return coreai_fn(x)

    _lower.__name__ = f"replace_{coreai_fn.__name__}"
    return _lower


# ---------------------------------------------------------------------------
# Bool round-trip helpers
#
# The Core AI runtime (ANE/MPS) crashes when a bool tensor that originates
# from a graph input is fed directly to coreai.not_() or
# coreai.broadcasting_and/or/xor().  Decompose through float32 to avoid the
# issue.  Internal bool tensors produced by comparison ops (e.g. the result
# of broadcasting_greater inside GreaterOrEqual/LessOrEqual) are safe because
# they never flow from a graph input boundary.
# ---------------------------------------------------------------------------


def _to_f32(x: Value) -> Value:
    return coreai.cast(x, np.float32)


def _cmp_operand(x: Value) -> Value:
    """Widen f16/bf16 comparison operands to float32 (value-exact).

    The Core AI runtime fails to load programs that compare half-precision
    graph inputs directly (Program load failure 0x10004, or an uncatchable
    MPSGraph abort when the bool result is a graph output).  f32 comparisons
    work and the widening cast is exact, so comparisons cast first.
    """
    if isinstance(tensor_type(x).element_type, F16Type | BF16Type):
        return _to_f32(x)
    return x


def _bool_result(x: Value) -> Value:
    """Cast float32 result back to bool via !=0."""
    return coreai.cast(
        coreai.broadcasting_not_equal(x, coreai.constant(0.0, dtype=np.float32)),
        np.bool_,
    )


# ---------------------------------------------------------------------------
# Sign helper (reused by Mod fmod=1 and Task 6 Sign op)
# ---------------------------------------------------------------------------


def _sign_value(x: Value) -> Value:
    """sign(x) = cast(x > 0, elem_type) - cast(x < 0, elem_type); NaN -> NaN."""
    e_type = tensor_type(x).element_type
    zero = coreai.constant(0, dtype=e_type)
    gt = coreai.cast(coreai.broadcasting_greater(x, zero), e_type)
    lt = coreai.cast(coreai.broadcasting_greater(zero, x), e_type)
    sign = coreai.broadcasting_sub(gt, lt)
    if isinstance(e_type, FloatType):
        # Both comparisons are False for NaN, giving 0; np.sign/onnxruntime
        # return NaN, so propagate the input NaN through a where().
        xc = _cmp_operand(x)
        sign = coreai.broadcasting_where(coreai.broadcasting_not_equal(xc, xc), x, sign)
    return sign


# ---------------------------------------------------------------------------
# Float-floor helper (used by Floor, Ceil, Round lowerings)
#
# coreai.broadcasting_floor_divide(x, 1.0) is NOT a floor on floats at
# runtime — the broadcasting decomposable op resolves to plain divide.
# Use: trunc-via-int32-cast, then subtract 1 for negative non-integers.
# ---------------------------------------------------------------------------


def _floor_f(x: Value) -> Value:
    """floor(x) for floating-point tensors via trunc + correction.

    Guard: float32 values with |x| >= 2^31 cannot be represented in int32,
    causing saturation to ±2147483647 when cast.  However, any float32 with
    |x| >= 2^24 is already an exact integer (the mantissa has no fractional
    bits), so floor(x) == x there.  We therefore replace the saturated trunc
    with x itself for out-of-range values.
    """
    e = tensor_type(x).element_type
    trunc_raw = coreai.cast(coreai.cast(x, np.int32), e)
    # Out-of-range floats are integral; replace saturated cast result with x.
    threshold = coreai.constant(float(2**31), dtype=e)
    in_range = coreai.broadcasting_greater(threshold, coreai.abs_(x))
    trunc = coreai.broadcasting_where(in_range, trunc_raw, x)
    zero = coreai.constant(0, dtype=e)
    one = coreai.constant(1, dtype=e)
    x_lt_zero = coreai.broadcasting_greater(zero, x)
    not_integer = coreai.broadcasting_not_equal(x, trunc)
    # Both booleans originate from comparisons — broadcasting_and is safe.
    need_correction = coreai.broadcasting_and(x_lt_zero, not_integer)
    return coreai.broadcasting_where(
        need_correction, coreai.broadcasting_sub(trunc, one), trunc
    )


# ---------------------------------------------------------------------------
# Softplus  — numerically stable: max(x,0) + log(1 + exp(-|x|))
#
# The naive form log(exp(x)+1) overflows to inf for x >= ~89 in float32.
# The stable form is mathematically equivalent:
#   softplus(x) = log(1 + exp(x))
#              = max(x,0) + log(exp(-max(x,0)) + exp(x - max(x,0)))
#              = max(x,0) + log(1 + exp(-|x|))
# ---------------------------------------------------------------------------


def _stable_softplus(x: Value) -> Value:
    """Numerically stable softplus: max(x,0) + log(1 + exp(-|x|))."""
    e = tensor_type(x).element_type
    zero = coreai.constant(0, dtype=e)
    one = coreai.constant(1, dtype=e)
    pos = coreai.broadcasting_maximum(x, zero)
    neg_abs_x = coreai.broadcasting_mul(coreai.abs_(x), coreai.constant(-1, dtype=e))
    return coreai.broadcasting_add(
        pos, coreai.log(coreai.broadcasting_add(one, coreai.exp(neg_abs_x)))
    )
