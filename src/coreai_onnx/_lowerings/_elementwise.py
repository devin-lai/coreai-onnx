# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for elementwise ONNX ops: binary/compare/logical, unary math, activations."""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import FloatType, IntegerType, Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, normalize_axis, operand, operands
from ._common import (
    _binary,
    _bool_result,
    _cmp_operand,
    _floor_f,
    _sign_value,
    _stable_softplus,
    _to_f32,
    _unary,
    _variadic_reduce,
)

# ---------------------------------------------------------------------------
# Pow  (base T and exponent T1 may differ; the result has the base type)
# ---------------------------------------------------------------------------


def replace_pow(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, y = operands(values_map, node, [0, 1])
    x_e, y_e = tensor_type(x).element_type, tensor_type(y).element_type
    if x_e == y_e:
        return coreai.broadcasting_pow(x, y)
    if isinstance(x_e, IntegerType) and isinstance(y_e, FloatType):
        # Casting a float exponent to int truncates it (0.5 -> 0).  The ONNX
        # reference computes np.power in float and casts the result back to
        # the base type.
        return coreai.cast(coreai.broadcasting_pow(coreai.cast(x, y_e), y), x_e)
    return coreai.broadcasting_pow(x, coreai.cast(y, x_e))


# ---------------------------------------------------------------------------
# Comparison ops
#
# f16/bf16 operands are widened to f32 via _cmp_operand (the runtime cannot
# load half-precision comparisons; the cast is value-exact).  >= and <= are
# built as (>) OR (==): a NOT-based rewrite would return True when an operand
# is NaN, where IEEE 754 requires every ordered comparison to be False.
# ---------------------------------------------------------------------------


def replace_equal(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    a, b = operands(values_map, node, [0, 1])
    e = tensor_type(a).element_type
    if isinstance(e, IntegerType) and e.width == 1:
        # bool graph inputs fed straight into a comparison crash the runtime
        # (same class as And/Or/Xor/Not, see _common.py); go through float32.
        return coreai.cast(coreai.broadcasting_equal(_to_f32(a), _to_f32(b)), np.bool_)
    return coreai.broadcasting_equal(_cmp_operand(a), _cmp_operand(b))


def replace_greater(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    a, b = operands(values_map, node, [0, 1])
    return coreai.broadcasting_greater(_cmp_operand(a), _cmp_operand(b))


def replace_greater_or_equal(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # a >= b  <=>  (a > b) OR (a == b)
    a, b = operands(values_map, node, [0, 1])
    a, b = _cmp_operand(a), _cmp_operand(b)
    return coreai.broadcasting_or(
        coreai.broadcasting_greater(a, b), coreai.broadcasting_equal(a, b)
    )


def replace_less(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # a < b  <=>  b > a
    a, b = operands(values_map, node, [0, 1])
    return coreai.broadcasting_greater(_cmp_operand(b), _cmp_operand(a))


def replace_less_or_equal(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # a <= b  <=>  (b > a) OR (a == b)
    a, b = operands(values_map, node, [0, 1])
    a, b = _cmp_operand(a), _cmp_operand(b)
    return coreai.broadcasting_or(
        coreai.broadcasting_greater(b, a), coreai.broadcasting_equal(a, b)
    )


# ---------------------------------------------------------------------------
# Logical ops
#
# The Core AI runtime (ANE/MPS) crashes when a bool tensor that originates
# from a graph input is fed directly to coreai.not_() or
# coreai.broadcasting_and/or/xor().  Decompose through float32 to avoid the
# issue.  Internal bool tensors produced by comparison ops (e.g. the result
# of broadcasting_greater inside GreaterOrEqual/LessOrEqual) are safe because
# they never flow from a graph input boundary.
# ---------------------------------------------------------------------------


def replace_and(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # AND = (cast_f32(a) * cast_f32(b)) != 0
    a, b = operands(values_map, node, [0, 1])
    return _bool_result(coreai.broadcasting_mul(_to_f32(a), _to_f32(b)))


def replace_or(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # OR = (cast_f32(a) + cast_f32(b)) != 0
    a, b = operands(values_map, node, [0, 1])
    return _bool_result(coreai.broadcasting_add(_to_f32(a), _to_f32(b)))


def replace_xor(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # XOR on bools = not_equal; broadcasting_not_equal already returns bool so
    # no !=0 round-trip through float is needed — just cast back to bool.
    a, b = operands(values_map, node, [0, 1])
    return coreai.cast(
        coreai.broadcasting_not_equal(_to_f32(a), _to_f32(b)),
        np.bool_,
    )


def replace_bitwise_not(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    elem = tensor_type(x).element_type
    if not isinstance(elem, IntegerType) or elem.width == 1:
        raise ValueError(f"BitwiseNot: expected an integer tensor, got {elem}")
    dtype = _integer_numpy_dtype(elem, "BitwiseNot")
    all_ones = np.iinfo(dtype).max if elem.is_unsigned else -1
    return coreai.broadcasting_bitwise_xor(
        x, coreai.constant(np.array(all_ones, dtype=dtype))
    )


def _integer_numpy_dtype(elem: IntegerType, op_name: str) -> np.dtype:
    if elem.width == 8 and elem.is_signed:
        return np.dtype(np.int8)
    if elem.width == 8 and elem.is_unsigned:
        return np.dtype(np.uint8)
    if elem.width == 16 and elem.is_signed:
        return np.dtype(np.int16)
    if elem.width == 16 and elem.is_unsigned:
        return np.dtype(np.uint16)
    if elem.width == 32 and elem.is_signed:
        return np.dtype(np.int32)
    if elem.width == 32 and elem.is_unsigned:
        return np.dtype(np.uint32)
    raise ValueError(f"{op_name}: unsupported integer element type {elem}")


def replace_not(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # NOT = (1.0 - cast_f32(a)) != 0
    (a,) = operands(values_map, node, [0])
    result = coreai.broadcasting_sub(coreai.constant(1.0, dtype=np.float32), _to_f32(a))
    return _bool_result(result)


# ---------------------------------------------------------------------------
# Where
# ---------------------------------------------------------------------------


def replace_where(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    cond, a, b = operands(values_map, node, [0, 1, 2])
    return coreai.broadcasting_where(cond, a, b)


# ---------------------------------------------------------------------------
# Mean  (variadic elementwise average with numpy broadcasting)
# ---------------------------------------------------------------------------


def replace_mean(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    vals = [values_map[name] for name in node.input]
    if not vals:
        raise ValueError(f"node '{node.name}': no inputs")
    result = vals[0]
    for v in vals[1:]:
        result = coreai.broadcasting_add(result, v)
    if len(vals) == 1:
        return result
    return coreai.broadcasting_divide(
        result,
        coreai.constant(float(len(vals)), dtype=tensor_type(result).element_type),
    )


# ---------------------------------------------------------------------------
# Mod  (fmod=0: sign follows divisor; fmod=1: sign follows dividend)
# ---------------------------------------------------------------------------


def replace_mod(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    node_attrs = attrs(node)
    fmod = node_attrs.get("fmod", 0)
    x, y = operands(values_map, node, [0, 1])

    if fmod:
        # C-style fmod: coreai.broadcasting_modulo is the truncated-division
        # remainder with IEEE-754 C-fmod semantics for floats (ModuloOp
        # docstring).  A decomposition through float division is inexact for
        # large |x| (off by a whole |y| when |x|/|y| rounds across an integer
        # boundary) and yields NaN for fmod(x, inf).
        return coreai.broadcasting_modulo(x, y)
    e_type = tensor_type(x).element_type
    zero = coreai.constant(0, dtype=e_type)
    # Python/ONNX Mod (fmod=0): sign follows divisor.
    # The truncated remainder must come from the native modulo op: the
    # decomposition x - trunc_div(x, y) * y is cancelled to 0 by
    # AIProgram.optimize() when y is a constant (real-number algebra,
    # invalid for truncating integer division). Correct to floor-mod via:
    #   floor_mod = where(trunc_r != 0 AND sign(trunc_r) != sign(y),
    #                     trunc_r + y, trunc_r)
    trunc_r = coreai.broadcasting_modulo(x, y)
    # sign mismatch iff exactly one of (trunc_r, y) is negative
    r_neg = coreai.broadcasting_greater(zero, trunc_r)
    y_neg = coreai.broadcasting_greater(zero, y)
    sign_differs = coreai.broadcasting_not_equal(r_neg, y_neg)
    r_nonzero = coreai.broadcasting_not_equal(trunc_r, zero)
    needs_correction = coreai.broadcasting_and(r_nonzero, sign_differs)
    corrected = coreai.broadcasting_add(trunc_r, y)
    return coreai.broadcasting_where(needs_correction, corrected, trunc_r)


# ---------------------------------------------------------------------------
# Direct unary ops
# ---------------------------------------------------------------------------


replace_abs = _unary(coreai.abs_)
replace_sqrt = _unary(coreai.sqrt)
replace_exp = _unary(coreai.exp)
replace_log = _unary(coreai.log)
replace_erf = _unary(coreai.erf)
replace_relu = _unary(coreai.relu)
replace_sigmoid = _unary(coreai.sigmoid)
replace_tanh = _unary(coreai.tanh)


def replace_round(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    """ONNX Round = round-half-to-even (banker's rounding).

    coreai.round_ uses half-away-from-zero, so we decompose manually:
      lo = floor(x)
      hi = lo + 1
      frac = x - lo
      at_half = (frac == 0.5)
      lo_is_even = (lo % 2 == 0)
      result = where(at_half, where(lo_is_even, lo, hi),
                               where(frac < 0.5, lo, hi))
    """
    (x,) = operands(values_map, node, [0])
    e = tensor_type(x).element_type
    lo = _floor_f(x)
    hi = coreai.broadcasting_add(lo, coreai.constant(1, dtype=e))
    frac = coreai.broadcasting_sub(x, lo)
    # lo % 2: use floor(lo / 2) * 2; lo is integer-valued
    lo_over_2 = _floor_f(coreai.broadcasting_divide(lo, coreai.constant(2, dtype=e)))
    lo_mod_2 = coreai.broadcasting_sub(
        lo, coreai.broadcasting_mul(coreai.constant(2, dtype=e), lo_over_2)
    )
    lo_is_even = coreai.broadcasting_equal(lo_mod_2, coreai.constant(0, dtype=e))
    # at_half: frac == 0.5 exactly
    half = coreai.constant(0.5, dtype=e)
    at_half = coreai.broadcasting_equal(frac, half)
    # result when exactly at half boundary: pick the even neighbor
    result_at_half = coreai.broadcasting_where(lo_is_even, lo, hi)
    # result otherwise: standard rounding (< 0.5 → lo, >= 0.5 → hi)
    result_otherwise = coreai.broadcasting_where(
        coreai.broadcasting_greater(half, frac), lo, hi
    )
    return coreai.broadcasting_where(at_half, result_at_half, result_otherwise)


replace_sin = _unary(coreai.sin)
replace_cos = _unary(coreai.cos)
replace_tan = _unary(coreai.tan)
replace_asin = _unary(coreai.asin)
replace_acos = _unary(coreai.acos)
replace_atan = _unary(coreai.atan)
replace_sinh = _unary(coreai.sinh)
replace_cosh = _unary(coreai.cosh)
replace_asinh = _unary(coreai.asinh)
replace_acosh = _unary(coreai.acosh)
replace_atanh = _unary(coreai.atanh)


# ---------------------------------------------------------------------------
# Gelu  (ONNX opset 20; 'approximate' attr: "none"|"tanh")
# ---------------------------------------------------------------------------


def replace_gelu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    approx = node_attrs.get("approximate", "none")
    return coreai.gelu(x, approximate=approx)


# ---------------------------------------------------------------------------
# Neg  (x * -1)
# ---------------------------------------------------------------------------


def replace_neg(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return coreai.broadcasting_mul(
        x, coreai.constant(-1, dtype=tensor_type(x).element_type)
    )


# ---------------------------------------------------------------------------
# Floor  (trunc + correction for negative non-integers)
# ---------------------------------------------------------------------------


def replace_floor(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return _floor_f(x)


# ---------------------------------------------------------------------------
# Ceil  (-floor(-x))
# ---------------------------------------------------------------------------


def replace_ceil(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    neg_one = coreai.constant(-1, dtype=tensor_type(x).element_type)
    neg_x = coreai.broadcasting_mul(x, neg_one)
    floor_neg_x = _floor_f(neg_x)
    return coreai.broadcasting_mul(floor_neg_x, neg_one)


# ---------------------------------------------------------------------------
# Reciprocal  (1 / x)
# ---------------------------------------------------------------------------


def replace_reciprocal(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return coreai.broadcasting_divide(
        coreai.constant(1, dtype=tensor_type(x).element_type), x
    )


# ---------------------------------------------------------------------------
# Sign  (reuse _sign_value helper)
# ---------------------------------------------------------------------------


def replace_sign(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return _sign_value(x)


# ---------------------------------------------------------------------------
# Hardmax  (one-hot first maximum along axis)
# ---------------------------------------------------------------------------


def replace_hardmax(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    axis = normalize_axis(attrs(node).get("axis", -1), rank)
    dim = tensor_type(x).shape[axis]
    if dim < 0:
        raise ValueError("Hardmax: axis dimension must be static")
    arg = coreai.cast(coreai.argmax(x, axis), np.int32)
    pos_shape = [1] * rank
    pos_shape[axis] = dim
    positions = coreai.constant(np.arange(dim, dtype=np.int32).reshape(pos_shape))
    mask = coreai.broadcasting_equal(positions, arg)
    return coreai.cast(mask, tensor_type(x).element_type)


# ---------------------------------------------------------------------------
# IsNaN  (x != x)
# ---------------------------------------------------------------------------


def replace_isnan(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    x = _cmp_operand(x)
    return coreai.broadcasting_not_equal(x, x)


# ---------------------------------------------------------------------------
# IsInf  ((x==+inf)|(x==-inf), honoring detect_positive/detect_negative attrs)
# ---------------------------------------------------------------------------


def replace_isinf(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    x = _cmp_operand(x)
    node_attrs = attrs(node)
    detect_pos = node_attrs.get("detect_positive", 1)
    detect_neg = node_attrs.get("detect_negative", 1)

    inf_val = coreai.constant(float("inf"), dtype=tensor_type(x).element_type)
    neg_inf_val = coreai.constant(float("-inf"), dtype=tensor_type(x).element_type)

    if detect_pos and detect_neg:
        # both operands come from comparison ops — broadcasting_or is safe
        return coreai.broadcasting_or(
            coreai.broadcasting_equal(x, inf_val),
            coreai.broadcasting_equal(x, neg_inf_val),
        )
    if detect_pos:
        return coreai.broadcasting_equal(x, inf_val)
    if detect_neg:
        return coreai.broadcasting_equal(x, neg_inf_val)
    # Neither infinity flavour requested: always False with x's shape.
    # A scalar constant would have the wrong shape in the IR; use
    # x > x (always False, same shape/dtype as x comparisons) instead.
    return coreai.broadcasting_greater(x, x)


# ---------------------------------------------------------------------------
# Clip  (min/max are optional inputs 1 and 2; cast bounds to x elem type)
# ---------------------------------------------------------------------------


def replace_clip(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    lo = operand(values_map, node, 1)
    hi = operand(values_map, node, 2)
    result = x
    if lo is not None:
        lo_cast = (
            coreai.cast(lo, tensor_type(x).element_type)
            if tensor_type(lo).element_type != tensor_type(x).element_type
            else lo
        )
        result = coreai.broadcasting_maximum(result, lo_cast)
    if hi is not None:
        hi_cast = (
            coreai.cast(hi, tensor_type(x).element_type)
            if tensor_type(hi).element_type != tensor_type(x).element_type
            else hi
        )
        result = coreai.broadcasting_minimum(result, hi_cast)
    if result is not x and isinstance(tensor_type(x).element_type, FloatType):
        # ONNX Clip follows np.clip: NaN in, NaN out.  The runtime's
        # maximum/minimum return the bound for a NaN operand, so patch the
        # input NaNs back in.
        xc = _cmp_operand(x)
        result = coreai.broadcasting_where(
            coreai.broadcasting_not_equal(xc, xc), x, result
        )
    return result


# ---------------------------------------------------------------------------
# LeakyRelu  (where(x>0, x, alpha*x))
# ---------------------------------------------------------------------------


def replace_leaky_relu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    alpha = float(node_attrs.get("alpha", 0.01))
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    slope = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    x_gt_zero = coreai.broadcasting_greater(x, zero)
    return coreai.broadcasting_where(x_gt_zero, x, coreai.broadcasting_mul(slope, x))


# ---------------------------------------------------------------------------
# ThresholdedRelu  (where(x > alpha, x, 0))
# ---------------------------------------------------------------------------


def replace_thresholded_relu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    alpha = float(attrs(node).get("alpha", 1.0))
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    alpha_val = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    return coreai.broadcasting_where(coreai.broadcasting_greater(x, alpha_val), x, zero)


# ---------------------------------------------------------------------------
# Shrink  (where(x<-lambda, x+bias, where(x>lambda, x-bias, 0)))
# ---------------------------------------------------------------------------


def replace_shrink(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    lambd = float(node_attrs.get("lambd", 0.5))
    bias = float(node_attrs.get("bias", 0.0))
    elem = tensor_type(x).element_type
    zero = coreai.constant(0, dtype=elem)
    lambda_val = coreai.constant(lambd, dtype=elem)
    neg_lambda_val = coreai.constant(-lambd, dtype=elem)
    bias_val = coreai.constant(bias, dtype=elem)
    pos = coreai.broadcasting_sub(x, bias_val)
    neg = coreai.broadcasting_add(x, bias_val)
    return coreai.broadcasting_where(
        coreai.broadcasting_greater(x, lambda_val),
        pos,
        coreai.broadcasting_where(
            coreai.broadcasting_greater(neg_lambda_val, x), neg, zero
        ),
    )


# ---------------------------------------------------------------------------
# PRelu  (relu(x) - slope*relu(-x), slope is input 1 with broadcasting)
# ---------------------------------------------------------------------------


def replace_prelu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, slope = operands(values_map, node, [0, 1])
    slope_cast = (
        coreai.cast(slope, tensor_type(x).element_type)
        if tensor_type(slope).element_type != tensor_type(x).element_type
        else slope
    )
    neg_x = coreai.broadcasting_mul(
        x, coreai.constant(-1, dtype=tensor_type(x).element_type)
    )
    return coreai.broadcasting_sub(
        coreai.relu(x),
        coreai.broadcasting_mul(slope_cast, coreai.relu(neg_x)),
    )


# ---------------------------------------------------------------------------
# Elu  (where(x>0, x, alpha*(exp(x)-1)))
# ---------------------------------------------------------------------------


def replace_elu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    alpha = float(node_attrs.get("alpha", 1.0))
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    alpha_val = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    one = coreai.constant(1, dtype=tensor_type(x).element_type)
    x_gt_zero = coreai.broadcasting_greater(x, zero)
    exp_x_minus_one = coreai.broadcasting_sub(coreai.exp(x), one)
    return coreai.broadcasting_where(
        x_gt_zero, x, coreai.broadcasting_mul(alpha_val, exp_x_minus_one)
    )


# ---------------------------------------------------------------------------
# Selu  (gamma*(where(x>0, x, alpha*(exp(x)-1))))
# ONNX spec defaults: alpha=1.6732631..., gamma=1.0507010...
# ---------------------------------------------------------------------------


def replace_selu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    alpha = float(node_attrs.get("alpha", 1.67326319217681884765625))
    gamma = float(node_attrs.get("gamma", 1.05070102214813232421875))
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    alpha_val = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    gamma_val = coreai.constant(gamma, dtype=tensor_type(x).element_type)
    one = coreai.constant(1, dtype=tensor_type(x).element_type)
    x_gt_zero = coreai.broadcasting_greater(x, zero)
    exp_x_minus_one = coreai.broadcasting_sub(coreai.exp(x), one)
    inner = coreai.broadcasting_where(
        x_gt_zero, x, coreai.broadcasting_mul(alpha_val, exp_x_minus_one)
    )
    return coreai.broadcasting_mul(gamma_val, inner)


# ---------------------------------------------------------------------------
# Celu  (max(0,x) + min(0, alpha*(exp(x/alpha)-1)))
# ---------------------------------------------------------------------------


def replace_celu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    alpha = float(node_attrs.get("alpha", 1.0))
    alpha_val = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    one = coreai.constant(1, dtype=tensor_type(x).element_type)
    x_over_alpha = coreai.broadcasting_divide(x, alpha_val)
    exp_part = coreai.broadcasting_sub(coreai.exp(x_over_alpha), one)
    negative_part = coreai.broadcasting_mul(alpha_val, exp_part)
    return coreai.broadcasting_add(
        coreai.broadcasting_maximum(x, zero),
        coreai.broadcasting_minimum(zero, negative_part),
    )


# ---------------------------------------------------------------------------
# HardSigmoid  (clip(alpha*x + beta, 0, 1))
# ---------------------------------------------------------------------------


def replace_hardsigmoid(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    alpha = float(node_attrs.get("alpha", 0.2))
    beta = float(node_attrs.get("beta", 0.5))
    alpha_val = coreai.constant(alpha, dtype=tensor_type(x).element_type)
    beta_val = coreai.constant(beta, dtype=tensor_type(x).element_type)
    zero = coreai.constant(0.0, dtype=tensor_type(x).element_type)
    one = coreai.constant(1.0, dtype=tensor_type(x).element_type)
    linear = coreai.broadcasting_add(coreai.broadcasting_mul(alpha_val, x), beta_val)
    return coreai.broadcasting_minimum(coreai.broadcasting_maximum(linear, zero), one)


# ---------------------------------------------------------------------------
# HardSwish  (x * hardsigmoid(x, alpha=1/6, beta=0.5))
# ---------------------------------------------------------------------------


def replace_hardswish(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    alpha_val = coreai.constant(1.0 / 6.0, dtype=tensor_type(x).element_type)
    beta_val = coreai.constant(0.5, dtype=tensor_type(x).element_type)
    zero = coreai.constant(0.0, dtype=tensor_type(x).element_type)
    one = coreai.constant(1.0, dtype=tensor_type(x).element_type)
    linear = coreai.broadcasting_add(coreai.broadcasting_mul(alpha_val, x), beta_val)
    hs = coreai.broadcasting_minimum(coreai.broadcasting_maximum(linear, zero), one)
    return coreai.broadcasting_mul(x, hs)


# ---------------------------------------------------------------------------
# Softmax  (axis attr default -1; coreai.softmax supports negative axes)
#
# There is no coreai.log_softmax primitive, so LogSoftmax is implemented
# below via the manual stable decomposition.
# ---------------------------------------------------------------------------


def replace_softmax(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    axis = node_attrs.get("axis", -1)
    # normalize negative axis
    axis = normalize_axis(axis, tensor_type(x).rank)
    return coreai.softmax(x, axis)


# ---------------------------------------------------------------------------
# LogSoftmax  (numerically stable: x - max(x) - log(sum(exp(x - max(x)))))
# ---------------------------------------------------------------------------


def replace_log_softmax(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    axis = node_attrs.get("axis", -1)
    axis = normalize_axis(axis, tensor_type(x).rank)
    axis_val = np.array([axis], dtype=np.int32)
    # reduce_max keeps the reduced dim as size 1 (ReduceMaxOp IR def;
    # the Python wrapper docstring is wrong)
    max_x = coreai.reduce_max(x, axis_val)
    x_shifted = coreai.broadcasting_sub(x, max_x)
    exp_shifted = coreai.exp(x_shifted)
    sum_exp = coreai.reduce_sum(exp_shifted, axis_val)
    log_sum_exp = coreai.log(sum_exp)
    return coreai.broadcasting_sub(x_shifted, log_sum_exp)


def replace_softplus(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return _stable_softplus(x)


# ---------------------------------------------------------------------------
# Softsign  (x / (1 + |x|))
# ---------------------------------------------------------------------------


def replace_softsign(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    one = coreai.constant(1, dtype=tensor_type(x).element_type)
    return coreai.broadcasting_divide(x, coreai.broadcasting_add(one, coreai.abs_(x)))


# ---------------------------------------------------------------------------
# Mish  (x * tanh(softplus(x)))
# ---------------------------------------------------------------------------


def replace_mish(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    softplus_x = _stable_softplus(x)
    return coreai.broadcasting_mul(x, coreai.tanh(softplus_x))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    # Arithmetic
    "Add": _binary(coreai.broadcasting_add),
    "Sub": _binary(coreai.broadcasting_sub),
    "Mul": _binary(coreai.broadcasting_mul),
    "Div": _binary(coreai.broadcasting_divide),
    "Pow": replace_pow,
    # Variadic elementwise
    "Min": _variadic_reduce(coreai.broadcasting_minimum, propagate_nan=True),
    "Max": _variadic_reduce(coreai.broadcasting_maximum, propagate_nan=True),
    "Mean": replace_mean,
    # Mod
    "Mod": replace_mod,
    # Comparison
    "Equal": replace_equal,
    "Greater": replace_greater,
    "GreaterOrEqual": replace_greater_or_equal,
    "Less": replace_less,
    "LessOrEqual": replace_less_or_equal,
    # Logical (decomposed through float32; direct primitives crash on bool inputs)
    "And": replace_and,
    "Or": replace_or,
    "Xor": replace_xor,
    "Not": replace_not,
    # Integer bitwise
    "BitwiseAnd": _binary(coreai.broadcasting_bitwise_and),
    "BitwiseOr": _binary(coreai.broadcasting_bitwise_or),
    "BitwiseXor": _binary(coreai.broadcasting_bitwise_xor),
    "BitwiseNot": replace_bitwise_not,
    # Where
    "Where": replace_where,
    # Unary math
    "Abs": replace_abs,
    "Sqrt": replace_sqrt,
    "Exp": replace_exp,
    "Log": replace_log,
    "Erf": replace_erf,
    "Relu": replace_relu,
    "Sigmoid": replace_sigmoid,
    "Tanh": replace_tanh,
    "Round": replace_round,
    "Gelu": replace_gelu,
    "Sin": replace_sin,
    "Cos": replace_cos,
    "Tan": replace_tan,
    "Asin": replace_asin,
    "Acos": replace_acos,
    "Atan": replace_atan,
    "Sinh": replace_sinh,
    "Cosh": replace_cosh,
    "Asinh": replace_asinh,
    "Acosh": replace_acosh,
    "Atanh": replace_atanh,
    # Decomposed unary
    "Neg": replace_neg,
    "Floor": replace_floor,
    "Ceil": replace_ceil,
    "Reciprocal": replace_reciprocal,
    "Sign": replace_sign,
    "Hardmax": replace_hardmax,
    "IsNaN": replace_isnan,
    "IsInf": replace_isinf,
    "Clip": replace_clip,
    # Activations
    "LeakyRelu": replace_leaky_relu,
    "ThresholdedRelu": replace_thresholded_relu,
    "Shrink": replace_shrink,
    "PRelu": replace_prelu,
    "Elu": replace_elu,
    "Selu": replace_selu,
    "Celu": replace_celu,
    "HardSigmoid": replace_hardsigmoid,
    "HardSwish": replace_hardswish,
    "Softmax": replace_softmax,
    "LogSoftmax": replace_log_softmax,
    "Softplus": replace_softplus,
    "Softsign": replace_softsign,
    "Mish": replace_mish,
}
