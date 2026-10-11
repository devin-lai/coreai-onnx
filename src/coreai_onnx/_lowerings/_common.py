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
    IntegerType,
    Location,
    OpResult,
    Type,
    Value,
    operand_values,
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
# Integer division  (ONNX Div on integers truncates toward zero)
#
# coreai's divide does truncate integers at runtime, but the pre-compilation
# rewrite simplifies it as real-number division: with a constant c,
# (x / c) * c becomes x, x - (x / c) * c becomes 0, and (x * a) / b becomes
# x * (a / b). floor_divide is left alone, so the quotient is built from it.
#
# The division is done on -|x| and -|y|: their quotient is non-negative, where
# flooring and truncating agree (the runtime has been seen doing either for
# negative integers), and unlike |x| a negated magnitude cannot overflow at the
# type minimum (abs(-128) is still -128 in int8). The sign goes back on after.
# ---------------------------------------------------------------------------


def _int_trunc_divide(x: Value, y: Value) -> Value:
    """x / y for integer tensors, truncating toward zero."""
    e = tensor_type(x).element_type
    if isinstance(e, IntegerType) and e.is_unsigned:
        return coreai.broadcasting_floor_divide(x, y)
    zero = coreai.constant(0, dtype=e)
    neg_one = coreai.constant(-1, dtype=e)

    def negated_magnitude(v: Value) -> Value:
        positive = coreai.broadcasting_greater(v, zero)
        return coreai.broadcasting_where(
            positive, coreai.broadcasting_mul(v, neg_one), v
        )

    magnitude = coreai.broadcasting_floor_divide(
        negated_magnitude(x), negated_magnitude(y)
    )
    negative = coreai.broadcasting_not_equal(
        coreai.broadcasting_greater(zero, x), coreai.broadcasting_greater(zero, y)
    )
    return coreai.broadcasting_where(
        negative, coreai.broadcasting_mul(magnitude, neg_one), magnitude
    )


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
# Float floor / ceil (used by the Floor and Ceil lowerings)
#
# floor(x) is floor_divide(x, 1), as in coreai-torch. It must not be built
# from a float -> int32 -> float cast round trip: Core AI's pre-compilation
# rewrite (and, on cpu_only, the BNNS backend) fuses that pair into a no-op,
# which turns floor into the identity.
#
# Half-precision operands are widened to f32 around the arithmetic (exact):
# f16 floor_divide fails to load on the default compute unit, and the ANE
# drops the outer negation of an f16 ceil.
# ---------------------------------------------------------------------------


def _half_as_f32(x: Value, fn: Callable[[Value], Value]) -> Value:
    """Apply *fn* to x, computing f16/bf16 inputs in f32 and narrowing back."""
    wide = _cmp_operand(x)
    result = fn(wide)
    if wide is x:
        return result
    return coreai.cast(result, tensor_type(x).element_type)


def _floor_divide_by_one(x: Value) -> Value:
    return coreai.broadcasting_floor_divide(
        x, coreai.constant(1, dtype=tensor_type(x).element_type)
    )


def _floor_f(x: Value) -> Value:
    """floor(x) for floating-point tensors."""
    return _half_as_f32(x, _floor_divide_by_one)


def _ceil_f(x: Value) -> Value:
    """ceil(x) = -floor(-x) for floating-point tensors."""

    def ceil(v: Value) -> Value:
        neg_one = coreai.constant(-1, dtype=tensor_type(v).element_type)
        floored = _floor_divide_by_one(coreai.broadcasting_mul(v, neg_one))
        return coreai.broadcasting_mul(floored, neg_one)

    return _half_as_f32(x, ceil)


def _trunc_f(x: Value) -> Value:
    """trunc(x): round toward zero, for floating-point tensors."""

    def trunc(v: Value) -> Value:
        e = tensor_type(v).element_type
        magnitude = _floor_divide_by_one(coreai.abs_(v))
        negative = coreai.broadcasting_greater(coreai.constant(0, dtype=e), v)
        return coreai.broadcasting_where(
            negative,
            coreai.broadcasting_mul(magnitude, coreai.constant(-1, dtype=e)),
            magnitude,
        )

    return _half_as_f32(x, trunc)


# ---------------------------------------------------------------------------
# Cast
#
# A cast that undoes the previous one (A -> B -> A) is fused into a no-op even
# when the pair changes values: Core AI's pre-compilation rewrite does it for
# float -> int32 -> float, and the cpu_only (BNNS) backend for every such pair
# it sees, bool included. Two shapes of ONNX Cast are therefore never emitted
# as a bare coreai.cast:
#   * to bool          -> x != 0, a comparison (ONNX: any nonzero is True)
#   * int back to float, where the int was itself cast from a float
#                      -> trunc() of that float, the value the pair computes
#     (values outside the integer type's range are undefined in ONNX).
# ---------------------------------------------------------------------------


def _is_bool(element_type: Type) -> bool:
    return isinstance(element_type, IntegerType) and element_type.width == 1


def _truncated_float(x: Value) -> Value | None:
    """The float value x was cast from, if x is the result of float -> int."""
    if not isinstance(x, OpResult) or x.owner.name != "coreai.cast":
        return None
    source = operand_values(x.owner)[0]
    if isinstance(tensor_type(source).element_type, FloatType):
        return source
    return None


def _cast(x: Value, target: Type) -> Value:
    """ONNX Cast of x to the Core AI element type *target*."""
    source = tensor_type(x).element_type
    if source == target:
        return x
    if _is_bool(target):
        operand = _cmp_operand(x)
        zero = coreai.constant(0, dtype=tensor_type(operand).element_type)
        return coreai.broadcasting_not_equal(operand, zero)
    if isinstance(target, FloatType) and isinstance(source, IntegerType):
        truncated = _truncated_float(x)
        if truncated is not None:
            return _cast(_trunc_f(truncated), target)
    return coreai.cast(x, target)


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
