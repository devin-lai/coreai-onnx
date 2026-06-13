# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for normalization ops, Dropout, and Resize."""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import (
    BF16Type,
    F16Type,
    F32Type,
    Location,
    RankedTensorType,
    Value,
    tensor_type,
)
from .._ir import coreai_dialect as coreai
from .._utils import attrs, normalize_axis, operand, operands
from ._common import _const_array, _require_const
from ._indexing import _take

# ---------------------------------------------------------------------------
# Normalizations
#
# All four follow the same pattern: (x - mean) * rsqrt(var + eps) * scale + b,
# differing only in where mean/var come from and how scale/b broadcast.
# coreai.reduce_mean keeps the reduced dims as size 1, so the statistics
# broadcast against x without reshaping.
# ---------------------------------------------------------------------------


def _per_channel(v: Value, rank: int) -> Value:
    """Reshape a [C] parameter to [C, 1, ...] so it broadcasts over NC(spatial)."""
    return coreai.reshape(v, [-1] + [1] * (rank - 2))


def _normalize(x: Value, dims: list[int], eps: float) -> Value:
    """(x - mean(x, dims)) * rsqrt(var(x, dims) + eps), dims kept."""
    mean = coreai.reduce_mean(x, dims)
    centered = coreai.broadcasting_sub(x, mean)
    var = coreai.reduce_mean(coreai.broadcasting_mul(centered, centered), dims)
    eps_c = coreai.constant(float(eps), dtype=tensor_type(x).element_type)
    return coreai.broadcasting_mul(
        centered, coreai.rsqrt(coreai.broadcasting_add(var, eps_c))
    )


def _normalize_stashed(
    x: Value, dims: list[int], eps: float, node_attrs: dict, op_name: str
) -> Value:
    """_normalize honoring ONNX stash_type: stage-one statistics default to
    f32, so f16/bf16 inputs are normalized in f32 and cast back."""
    stash = node_attrs.get("stash_type", onnx.TensorProto.FLOAT)
    et = tensor_type(x).element_type
    if stash == onnx.TensorProto.FLOAT and (
        F16Type.isinstance(et) or BF16Type.isinstance(et)
    ):
        norm = _normalize(coreai.cast(x, np.float32), dims, eps)
        return coreai.cast(norm, RankedTensorType.get(tensor_type(norm).shape, et))
    expected = {
        onnx.TensorProto.FLOAT: F32Type,
        onnx.TensorProto.FLOAT16: F16Type,
        onnx.TensorProto.BFLOAT16: BF16Type,
    }.get(stash)
    if expected is None or not expected.isinstance(et):
        raise ValueError(
            f"{op_name}: stash_type={stash} with input type {et} is not supported"
        )
    return _normalize(x, dims, eps)


def replace_batch_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    node_attrs = attrs(node)
    if node_attrs.get("training_mode", 0):
        raise ValueError(
            "BatchNormalization: training_mode=1 is not supported (inference only)"
        )
    x, scale, b, mean, var = operands(values_map, node, [0, 1, 2, 3, 4])
    rank = tensor_type(x).rank
    if rank < 2:
        raise ValueError("BatchNormalization: input must be at least rank 2")
    eps = float(node_attrs.get("epsilon", 1e-5))
    # Canonical y = x*A + B (A = scale*rsqrt(var+eps), B = b - mean*A): two
    # full-tensor ops instead of four, and the conv -> mul(const) -> add(const)
    # form is what the compiler's FuseConvAndScalingOp folds into conv weights.
    a: Value
    bias: Value
    sc, bc, mc, vc = (_const_array(v) for v in (scale, b, mean, var))
    if sc is not None and bc is not None and mc is not None and vc is not None:
        s64, b64, m64, v64 = (c.astype(np.float64) for c in (sc, bc, mc, vc))
        a64 = s64 / np.sqrt(v64 + eps)
        a = coreai.constant(a64.astype(sc.dtype))
        bias = coreai.constant((b64 - m64 * a64).astype(bc.dtype))
    else:
        eps_c = coreai.constant(eps, dtype=tensor_type(x).element_type)
        a = coreai.broadcasting_mul(
            scale, coreai.rsqrt(coreai.broadcasting_add(var, eps_c))
        )
        bias = coreai.broadcasting_sub(b, coreai.broadcasting_mul(mean, a))
    return coreai.broadcasting_add(
        coreai.broadcasting_mul(x, _per_channel(a, rank)),
        _per_channel(bias, rank),
    )


def replace_instance_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, scale, b = operands(values_map, node, [0, 1, 2])
    rank = tensor_type(x).rank
    if rank < 3:
        raise ValueError(
            "InstanceNormalization: input must be at least rank 3 (N, C, spatial...)"
        )
    eps = attrs(node).get("epsilon", 1e-5)
    norm = _normalize(x, list(range(2, rank)), eps)
    return coreai.broadcasting_add(
        coreai.broadcasting_mul(norm, _per_channel(scale, rank)),
        _per_channel(b, rank),
    )


def replace_layer_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    if len([o for o in node.output if o]) > 1:
        raise ValueError("LayerNormalization: Mean/InvStdDev outputs are not supported")
    x, scale = operands(values_map, node, [0, 1])
    bias = operand(values_map, node, 2)
    rank = tensor_type(x).rank
    node_attrs = attrs(node)
    axis = normalize_axis(node_attrs.get("axis", -1), rank)
    eps = node_attrs.get("epsilon", 1e-5)
    norm = _normalize_stashed(
        x, list(range(axis, rank)), eps, node_attrs, "LayerNormalization"
    )
    # Scale/bias cover the normalized (trailing) dims, so they broadcast as-is.
    result = coreai.broadcasting_mul(norm, scale)
    if bias is not None:
        result = coreai.broadcasting_add(result, bias)
    return result


def replace_group_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, scale, bias = operands(values_map, node, [0, 1, 2])
    node_attrs = attrs(node)
    groups = node_attrs["num_groups"]
    eps = node_attrs.get("epsilon", 1e-5)
    rank = tensor_type(x).rank
    if rank < 3:
        raise ValueError(
            "GroupNormalization: input must be at least rank 3 (N, C, spatial...)"
        )
    channels = tensor_type(x).shape[1]
    if channels < 0:
        raise ValueError("GroupNormalization: requires a static channel dimension")
    if channels % groups != 0:
        raise ValueError(
            f"GroupNormalization: C={channels} not divisible by num_groups={groups}"
        )
    # Reshape to [N, G, C/G, -1], normalize over (C/G, flattened spatial).
    batch_dim = coreai.cast(
        coreai.slice_(coreai.get_shape(x), [0], [1], [1]), dtype=np.int32
    )
    group_shape = coreai.constant(
        np.array([groups, channels // groups, -1], dtype=np.int32)
    )
    target = coreai.concat(0, [batch_dim, group_shape])
    norm = _normalize_stashed(
        coreai.reshape(x, target), [2, 3], eps, node_attrs, "GroupNormalization"
    )
    norm = coreai.reshape(norm, coreai.get_shape(x))
    # Opset 21+ semantics: per-channel scale/bias of shape [C].
    return coreai.broadcasting_add(
        coreai.broadcasting_mul(norm, _per_channel(scale, rank)),
        _per_channel(bias, rank),
    )


def _lp_norm(x: Value, axis: int, p: int) -> Value:
    abs_x = coreai.abs_(x)
    if p == 1:
        return coreai.reduce_sum(abs_x, [axis])
    return coreai.sqrt(coreai.reduce_sum(coreai.broadcasting_mul(x, x), [axis]))


def replace_lp_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    p = int(node_attrs.get("p", 2))
    if p not in (1, 2):
        raise ValueError(f"LpNormalization: p={p} is not supported (supported: 1, 2)")
    axis = normalize_axis(node_attrs.get("axis", -1), tensor_type(x).rank)
    norm = _lp_norm(x, axis, p)
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    norm_is_zero = coreai.broadcasting_equal(norm, zero)
    quotient = coreai.broadcasting_divide(
        x,
        coreai.broadcasting_where(
            norm_is_zero, coreai.constant(1, dtype=tensor_type(x).element_type), norm
        ),
    )
    return coreai.broadcasting_where(norm_is_zero, zero, quotient)


def replace_mean_variance_normalization(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    axes = [normalize_axis(int(a), rank) for a in attrs(node).get("axes", [0, 2, 3])]
    if len(axes) != len(set(axes)):
        raise ValueError("MeanVarianceNormalization: duplicate axes")
    mean = coreai.reduce_mean(x, sorted(axes))
    centered = coreai.broadcasting_sub(x, mean)
    var = coreai.reduce_mean(coreai.broadcasting_mul(centered, centered), sorted(axes))
    return coreai.broadcasting_divide(centered, coreai.sqrt(var))


# ---------------------------------------------------------------------------
# Dropout  (inference mode: identity)
# ---------------------------------------------------------------------------


def replace_dropout(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    # A declared (non-empty) second output means the mask is consumed.
    if len(node.output) > 1 and node.output[1]:
        raise ValueError("Dropout: the mask output is not supported")
    training = operand(values_map, node, 2)
    if training is not None:
        t = _require_const(training, "Dropout", "'training_mode' input")
        if t.reshape(-1).astype(bool).any():
            raise ValueError(
                "Dropout: training_mode=true is not supported (inference only)"
            )
    (x,) = operands(values_map, node, [0])
    return x  # the 'ratio' input is irrelevant in inference mode


# ---------------------------------------------------------------------------
# Resize  (2D spatial NCHW, nearest/linear)
#
# Both modes are lowered as exact compile-time index computations instead of
# coreai.interpolate: nearest because the native kernel rounds ties-to-even,
# which differs from every ONNX nearest_mode at exact .5 ties (e.g.
# asymmetric x2); linear because the native resample kernel aborts the
# process at GPU/ANE compile time in real graphs (see replace_resize).
# ---------------------------------------------------------------------------

_COORD_MODES = ("half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric")


def _resize_coords(
    out_len: int, in_len: int, scale: float, coord_mode: str
) -> np.ndarray:
    """ONNX input-space coordinates for each output index (float64)."""
    i = np.arange(out_len, dtype=np.float64)
    if coord_mode == "half_pixel":
        return (i + 0.5) / scale - 0.5
    if coord_mode == "pytorch_half_pixel":
        return (i + 0.5) / scale - 0.5 if out_len > 1 else np.zeros(1)
    if coord_mode == "align_corners":
        if out_len <= 1:
            return np.zeros(1)
        return i * (in_len - 1) / (out_len - 1)
    if coord_mode != "asymmetric":
        raise ValueError(
            f"Resize: coordinate_transformation_mode '{coord_mode}' is not supported"
        )
    return i / scale


def _nearest_indices(
    out_len: int, in_len: int, scale: float, coord_mode: str, nearest_mode: str
) -> np.ndarray:
    c = _resize_coords(out_len, in_len, scale, coord_mode)
    if nearest_mode == "round_prefer_floor":
        idx = np.ceil(c - 0.5)
    elif nearest_mode == "round_prefer_ceil":
        idx = np.floor(c + 0.5)
    elif nearest_mode == "floor":
        idx = np.floor(c)
    elif nearest_mode == "ceil":
        idx = np.ceil(c)
    else:
        raise ValueError(f"Resize: nearest_mode '{nearest_mode}' is not supported")
    return np.clip(idx, 0, in_len - 1).astype(np.int32)


def _linear_axis_interp(
    x: Value, out_len: int, in_len: int, scale: float, coord_mode: str, axis: int
) -> Value:
    """1-D linear interpolation along *axis* as gather(lo) * (1 - w) + gather(hi) * w.

    Indices and weights are compile-time constants (ONNX reference semantics:
    neighbor indices clip to the edges, the fraction comes from the unclipped
    coordinate, so out-of-range coordinates degenerate to edge replication).
    """
    c = _resize_coords(out_len, in_len, scale, coord_mode)
    floor = np.floor(c)
    lo = np.clip(floor, 0, in_len - 1).astype(np.int32)
    hi = np.clip(floor + 1, 0, in_len - 1).astype(np.int32)
    frac = (c - floor).astype(np.float32)
    if np.all(frac == 0.0) and np.array_equal(lo, hi):
        # Pure index remap (e.g. identity or integer downscale): one gather.
        return _take(x, coreai.constant(lo), axis)
    weight_shape = [1, 1, 1, 1]
    weight_shape[axis] = out_len
    elem_type = tensor_type(x).element_type
    w_hi = coreai.constant(frac.reshape(weight_shape), dtype=elem_type)
    w_lo = coreai.constant((1.0 - frac).reshape(weight_shape), dtype=elem_type)
    lo_v = coreai.broadcasting_mul(_take(x, coreai.constant(lo), axis), w_lo)
    hi_v = coreai.broadcasting_mul(_take(x, coreai.constant(hi), axis), w_hi)
    return coreai.broadcasting_add(lo_v, hi_v)


def _optional_nonempty(v: Value | None) -> Value | None:
    """None for absent inputs and for the empty-tensor 'not provided' idiom."""
    if v is None or 0 in tensor_type(v).shape:
        return None
    return v


def replace_resize(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    node_attrs = attrs(node)
    mode = node_attrs.get("mode", "nearest")
    if mode not in ("nearest", "linear"):
        raise ValueError(f"Resize: mode '{mode}' is not supported")
    if node_attrs.get("antialias", 0):
        raise ValueError("Resize: antialias=1 is not supported")
    if node_attrs.get("exclude_outside", 0):
        raise ValueError("Resize: exclude_outside=1 is not supported")
    policy = node_attrs.get("keep_aspect_ratio_policy", "stretch")
    if policy != "stretch":
        raise ValueError(
            f"Resize: keep_aspect_ratio_policy '{policy}' is not supported"
        )
    coord_mode = node_attrs.get("coordinate_transformation_mode", "half_pixel")
    if coord_mode not in _COORD_MODES:
        raise ValueError(
            f"Resize: coordinate_transformation_mode '{coord_mode}' is not supported"
        )

    (x,) = operands(values_map, node, [0])
    if tensor_type(x).rank != 4:
        raise ValueError("Resize: only rank-4 (NCHW) inputs are supported")
    n, c, in_h, in_w = tensor_type(x).shape
    if min(n, c, in_h, in_w) < 0:
        raise ValueError("Resize: requires a fully static input shape")

    axes: list[int] | None = node_attrs.get("axes")
    if axes is not None:
        axes = [normalize_axis(a, 4) for a in axes]
        if sorted(axes) != [2, 3]:
            raise ValueError("Resize: only spatial axes [2, 3] are supported")

    scales_v = _optional_nonempty(operand(values_map, node, 2))
    sizes_v = _optional_nonempty(operand(values_map, node, 3))
    if (scales_v is None) == (sizes_v is None):
        raise ValueError("Resize: exactly one of 'scales' or 'sizes' must be provided")

    if scales_v is not None:
        scales = _require_const(scales_v, "Resize", "'scales' input").astype(np.float64)
        if axes is None:
            if scales.shape != (4,) or scales[0] != 1.0 or scales[1] != 1.0:
                raise ValueError(
                    "Resize: only spatial resize is supported "
                    "(batch/channel scales must be 1)"
                )
            scales = scales[2:]
        else:
            # Opset 18+: per-axis values are given in the order of `axes`.
            scales = scales[np.argsort(axes)]
        scale_h, scale_w = float(scales[0]), float(scales[1])
        out_h = int(np.floor(in_h * scale_h))
        out_w = int(np.floor(in_w * scale_w))
    else:
        if sizes_v is None:
            raise ValueError(
                "Resize: exactly one of 'scales' or 'sizes' must be provided"
            )
        sizes = _require_const(sizes_v, "Resize", "'sizes' input").astype(np.int64)
        if axes is None:
            if sizes.shape != (4,) or sizes[0] != n or sizes[1] != c:
                raise ValueError(
                    "Resize: only spatial resize is supported "
                    "(batch/channel sizes must match the input)"
                )
            sizes = sizes[2:]
        else:
            sizes = sizes[np.argsort(axes)]  # same axes-order rule as scales
        out_h, out_w = int(sizes[0]), int(sizes[1])
        scale_h, scale_w = out_h / in_h, out_w / in_w

    if mode == "nearest":
        nearest_mode = node_attrs.get("nearest_mode", "round_prefer_floor")
        idx_h = _nearest_indices(out_h, in_h, scale_h, coord_mode, nearest_mode)
        idx_w = _nearest_indices(out_w, in_w, scale_w, coord_mode, nearest_mode)
        result = _take(x, coreai.constant(idx_h), 2)
        return _take(result, coreai.constant(idx_w), 3)

    # Linear mode is decomposed into exact compile-time gather+lerp along H
    # then W instead of coreai.interpolate: the native resample kernel the
    # interpolate lowers to (MPSNDArrayResample) aborts the whole process at
    # GPU/ANE compile time in real decoder graphs ("source and destination
    # channels mismatch", track_anything's segment model), and the failing
    # surrounding-op set is not cleanly characterizable - pad+slice barriers
    # around the interpolate do not prevent it. The decomposition mirrors how
    # nearest mode already avoids the kernel via _take.
    result = _linear_axis_interp(x, out_h, in_h, scale_h, coord_mode, axis=2)
    return _linear_axis_interp(result, out_w, in_w, scale_w, coord_mode, axis=3)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    "BatchNormalization": replace_batch_normalization,
    "InstanceNormalization": replace_instance_normalization,
    "LayerNormalization": replace_layer_normalization,
    "GroupNormalization": replace_group_normalization,
    "LpNormalization": replace_lp_normalization,
    "MeanVarianceNormalization": replace_mean_variance_normalization,
    "Dropout": replace_dropout,
    "Resize": replace_resize,
}
