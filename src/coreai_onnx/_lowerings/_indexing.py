# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for gather/scatter-family and indexing ONNX ops."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import Attribute, F32Type, Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._type_mapping import coreai_type_from_onnx_dtype
from .._utils import attrs, normalize_axis, operands
from ._common import _const_array, _require_const

# ---------------------------------------------------------------------------
# Gather / Scatter family + indexing ops
#
# ONNX indices are int64 (narrowed to si32 by the type layer) and may be
# negative; coreai gather/scatter primitives want non-negative int32, so
# every lowering normalizes: constant indices in numpy, runtime indices via
# where(idx < 0, idx + dim, idx).
# ---------------------------------------------------------------------------


def _normalize_indices(
    idx_v: Value,
    dims: np.ndarray | np.int64,
    op_name: str,
    x: Value,
    span: tuple[int, int],
) -> Value:
    """Non-negative int32 indices; *dims* (scalar or [d]) broadcasts over idx.

    *x* and *span* locate the indexed dims (x.shape[span[0]:span[1]]) so that
    dynamic entries in *dims* can be read via get_shape at runtime.
    """
    arr = _const_array(idx_v)
    static = not (np.asarray(dims) < 0).any()
    idx: Value
    if arr is not None:
        a = arr.astype(np.int64)
        if not (a < 0).any():
            return coreai.constant(a.astype(np.int32))
        if static:
            return coreai.constant(np.where(a < 0, a + dims, a).astype(np.int32))
        idx = coreai.constant(a.astype(np.int32))
    else:
        idx = coreai.cast(idx_v, np.int32)
    dims_v: Value
    if static:
        dims_v = coreai.constant(np.asarray(dims).astype(np.int32))
    else:
        b, e = span
        dims_v = coreai.cast(
            coreai.slice_(coreai.get_shape(x), [b], [e], [1]), np.int32
        )
        if np.ndim(dims) == 0:
            dims_v = coreai.reshape(dims_v, [])  # keep the scalar broadcast rank
    neg = coreai.broadcasting_greater(coreai.constant(0, dtype=np.int32), idx)
    shifted = coreai.broadcasting_add(idx, dims_v)
    return coreai.broadcasting_where(neg, shifted, idx)


def _take(x: Value, idx: Value, axis: int) -> Value:
    """numpy.take: gather slices of *x* along *axis* at *idx* (any rank, >= 0).

    Result shape: x.shape[:axis] ++ idx.shape ++ x.shape[axis+1:].  Lowered as
    transpose(axis to front) + gather_nd + transpose(index block into place).
    """
    rank = tensor_type(x).rank
    q = tensor_type(idx).rank
    if axis:
        front = [axis, *range(axis), *range(axis + 1, rank)]
        x = coreai.transpose(x, np.array(front, dtype=np.uint32))
    g = coreai.gather_nd(input=x, indices=coreai.expand_dims(idx, [q]))
    if axis and q != 0:
        # g dims: [idx (q), x[:axis] (axis), x[axis+1:]]; move idx block to axis.
        perm = [*range(q, q + axis), *range(q), *range(q + axis, q + rank - 1)]
        g = coreai.transpose(g, np.array(perm, dtype=np.uint32))
    return g


def replace_gather(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, idx_v = operands(values_map, node, [0, 1])
    axis = normalize_axis(attrs(node).get("axis", 0), tensor_type(x).rank)
    # np.int64: a dynamic dim reports as MLIR's kDynamic sentinel (INT64_MIN),
    # which overflows int32; the value only reaches IR on the static path.
    idx = _normalize_indices(
        idx_v, np.int64(tensor_type(x).shape[axis]), "Gather", x, (axis, axis + 1)
    )
    return _take(x, idx, axis)


def replace_gather_elements(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, idx_v = operands(values_map, node, [0, 1])
    axis = normalize_axis(attrs(node).get("axis", 0), tensor_type(x).rank)
    idx = _normalize_indices(
        idx_v,
        np.int64(tensor_type(x).shape[axis]),
        "GatherElements",
        x,
        (axis, axis + 1),
    )
    return coreai.gather_along_axis(x, idx, axis)


def replace_gather_nd(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, idx_v = operands(values_map, node, [0, 1])
    b = attrs(node).get("batch_dims", 0)
    d = tensor_type(idx_v).shape[-1]
    if d < 0:
        raise ValueError("GatherND: indices must have a static last dimension")
    dims = np.array(tensor_type(x).shape[b : b + d], dtype=np.int64)
    idx = _normalize_indices(idx_v, dims, "GatherND", x, (b, b + d))
    if b == 0:
        return coreai.gather_nd(input=x, indices=idx)

    # coreai.gather_nd has no batch_dims: flatten the batch dims to one and
    # prepend an explicit batch coordinate to each index tuple.
    batch = list(tensor_type(x).shape[:b])
    mid = list(tensor_type(idx_v).shape[b:-1])
    tail = list(tensor_type(x).shape[b:])
    if any(s < 0 for s in batch + mid + tail):
        raise ValueError("GatherND: batch_dims > 0 requires static shapes")
    n = math.prod(batch)
    x2 = coreai.reshape(x, [n, *tail])
    idx2 = coreai.reshape(idx, [n, *mid, d])
    coord = np.ascontiguousarray(
        np.broadcast_to(
            np.arange(n, dtype=np.int32).reshape([n] + [1] * (len(mid) + 1)),
            [n, *mid, 1],
        )
    )
    full_idx = coreai.concat(len(mid) + 1, [coreai.constant(coord), idx2])
    g = coreai.gather_nd(input=x2, indices=full_idx)
    return coreai.reshape(g, [*batch, *mid, *tail[d:]])


def _scatter_updates(upd: Value, x: Value) -> Value:
    if tensor_type(upd).element_type != tensor_type(x).element_type:
        return coreai.cast(upd, tensor_type(x).element_type)
    return upd


def replace_scatter_elements(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, idx_v, upd = operands(values_map, node, [0, 1, 2])
    node_attrs = attrs(node)
    reduction = node_attrs.get("reduction", "none")
    if reduction != "none":
        raise ValueError(
            f"ScatterElements: reduction='{reduction}' is not supported (only 'none')"
        )
    axis = normalize_axis(node_attrs.get("axis", 0), tensor_type(x).rank)
    # ONNX leaves duplicate-index behavior undefined for reduction="none";
    # the lowering passes indices through unmodified.
    idx = _normalize_indices(
        idx_v,
        np.int64(tensor_type(x).shape[axis]),
        "ScatterElements",
        x,
        (axis, axis + 1),
    )
    return coreai.scatter_along_axis(
        output=x.type, input=x, indices=idx, updates=_scatter_updates(upd, x), axis=axis
    )


def replace_scatter_nd(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, idx_v, upd = operands(values_map, node, [0, 1, 2])
    reduction = attrs(node).get("reduction", "none")
    if reduction != "none":
        raise ValueError(
            f"ScatterND: reduction='{reduction}' is not supported (only 'none')"
        )
    d = tensor_type(idx_v).shape[-1]
    if d < 0:
        raise ValueError("ScatterND: indices must have a static last dimension")
    dims = np.array(tensor_type(x).shape[:d], dtype=np.int64)
    # ONNX leaves duplicate-index behavior undefined for reduction="none";
    # the lowering passes indices through unmodified.
    idx = _normalize_indices(idx_v, dims, "ScatterND", x, (0, d))
    return coreai.scatter_nd(input=x, indices=idx, updates=_scatter_updates(upd, x))


def replace_one_hot(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    idx_v, depth_v, values_v = operands(values_map, node, [0, 1, 2])
    depth = int(_require_const(depth_v, "OneHot", "'depth' input").reshape(-1)[0])
    rank = tensor_type(idx_v).rank
    axis = normalize_axis(attrs(node).get("axis", -1), rank + 1)
    # Negative indices wrap once (idx + depth); anything still out of
    # [0, depth) never matches the iota and yields the off value, per spec.
    idx = coreai.cast(idx_v, np.int32)
    neg = coreai.broadcasting_greater(coreai.constant(0, dtype=np.int32), idx)
    wrapped = coreai.broadcasting_add(idx, coreai.constant(depth, dtype=np.int32))
    idx = coreai.broadcasting_where(neg, wrapped, idx)
    mask = coreai.broadcasting_equal(
        coreai.expand_dims(idx, [rank]),
        coreai.constant(np.arange(depth, dtype=np.int32)),
    )
    off = coreai.shrink_dims(coreai.slice_(values_v, [0], [1], [1]), [0])
    on = coreai.shrink_dims(coreai.slice_(values_v, [1], [2], [1]), [0])
    out = coreai.broadcasting_where(mask, on, off)
    if axis != rank:
        perm = [*range(axis), rank, *range(axis, rank)]
        out = coreai.transpose(out, np.array(perm, dtype=np.uint32))
    return out


def replace_non_zero(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    # coreai.non_zero yields (num_nonzero, rank); ONNX wants (rank, num_nonzero).
    nz = coreai.transpose(coreai.non_zero(x), np.array([1, 0], dtype=np.uint32))
    return coreai.cast(nz, np.int32)


def replace_eye_like(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    if tensor_type(x).rank != 2:
        raise ValueError("EyeLike: input must be rank 2")
    rows, cols = tensor_type(x).shape
    if rows < 0 or cols < 0:
        raise ValueError("EyeLike: requires a static input shape")
    node_attrs = attrs(node)
    eye = coreai.constant(
        np.eye(rows, cols, k=node_attrs.get("k", 0), dtype=np.float32)
    )
    dtype = node_attrs.get("dtype")
    target = (
        coreai_type_from_onnx_dtype(dtype)
        if dtype is not None
        else tensor_type(x).element_type
    )
    if tensor_type(eye).element_type == target:
        return eye
    return coreai.cast(eye, target)


def replace_compress(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, cond_v = operands(values_map, node, [0, 1])
    cond = _const_array(cond_v)
    if cond is None:
        raise ValueError(
            "Compress: 'condition' must be a compile-time constant "
            "(a runtime condition implies a dynamic output shape)"
        )
    idx = np.nonzero(cond.reshape(-1).astype(bool))[0].astype(np.int32)
    axis_attr = attrs(node).get("axis")
    if axis_attr is None:
        x = coreai.reshape(x, [-1])
        axis = 0
    else:
        axis = normalize_axis(axis_attr, tensor_type(x).rank)
    return _take(x, coreai.constant(idx), axis)


# ---------------------------------------------------------------------------
# GridSample  (2D spatial bilinear/nearest sampling at per-pixel grid coords)
#
# Lowered through coreai.interpolate_coordinates, which samples a K-dim field
# at [N, K] absolute-index coordinates (0 -> first element, dim-1 -> last) and
# clamps out-of-bounds reads to the edge. One rank-3 [C, H, W] field is sampled
# per batch with coords (channel, row, col); the channel index is an exact
# integer, so the K-linear interpolation reduces to bilinear in that channel's
# plane (the ceil neighbor's weight is zero).
#
# padding_mode:
#   border  -> the native clamp.
#   zeros   -> pad the field with one zero pixel per spatial edge and shift the
#              coords by +1; clamping to the (now zero) edge then reproduces the
#              'zeros' rule exactly, including the partial-weight border band.
# ---------------------------------------------------------------------------


def replace_grid_sample(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, grid = operands(values_map, node, [0, 1])
    node_attrs = attrs(node)
    mode = node_attrs.get("mode", "linear")
    padding_mode = node_attrs.get("padding_mode", "zeros")
    align_corners = int(node_attrs.get("align_corners", 0))

    if tensor_type(x).rank != 4:
        raise ValueError("GridSample: only rank-4 (NCHW) input is supported")
    nb, c, in_h, in_w = tensor_type(x).shape
    grid_shape = tensor_type(grid).shape
    if tensor_type(grid).rank != 4 or grid_shape[-1] != 2:
        raise ValueError("GridSample: grid must have shape [N, H_out, W_out, 2]")
    out_h, out_w = grid_shape[1], grid_shape[2]
    if min(nb, c, in_h, in_w) < 0 or out_h < 0 or out_w < 0 or grid_shape[0] != nb:
        raise ValueError(
            "GridSample: requires fully static input/grid shapes with a "
            "matching batch size"
        )

    if mode in ("bilinear", "linear"):
        interp = "linear"
    elif mode == "nearest":
        interp = "nearest_neighbor"
    else:
        raise ValueError(
            f"GridSample: mode '{mode}' is not supported (bilinear/linear, nearest)"
        )
    if padding_mode not in ("zeros", "border"):
        raise ValueError(
            f"GridSample: padding_mode '{padding_mode}' is not supported "
            "(zeros, border)"
        )
    mode_attr = Attribute.parse(f"#coreai.interpolation_mode<{interp}>")

    # Sample (and compute coordinates) in float32; cast a non-f32 field back.
    elem = tensor_type(x).element_type
    f32 = isinstance(elem, F32Type)
    xf = x if f32 else coreai.cast(x, np.float32)
    gf = (
        grid
        if isinstance(tensor_type(grid).element_type, F32Type)
        else coreai.cast(grid, np.float32)
    )

    # 'zeros' padding pads one zero pixel per spatial edge *per batch* (below):
    # padding the whole [N, C, H, W] field at once materializes an N*C*H*W
    # tensor that overflows the GPU's MTLBuffer-aliasing limit for ROI-pooling
    # batches (e.g. detectron2's 200x1024x50x50). shift moves coords past it.
    shift = 1.0 if padding_mode == "zeros" else 0.0

    def denorm(comp: Value, size: int) -> Value:
        # grid value g in [-1, 1] -> absolute input-pixel index (+shift for pad).
        if align_corners:
            scale, bias = 0.5 * (size - 1), 0.5 * (size - 1) + shift
        else:
            scale, bias = 0.5 * size, 0.5 * size - 0.5 + shift
        return coreai.broadcasting_add(
            coreai.broadcasting_mul(comp, coreai.constant(scale, dtype=np.float32)),
            coreai.constant(bias, dtype=np.float32),
        )

    # grid[..., 0] indexes width, grid[..., 1] indexes height.
    gx = coreai.shrink_dims(
        coreai.slice_(gf, [0, 0, 0, 0], [nb, out_h, out_w, 1], [1, 1, 1, 1]), [3]
    )
    gy = coreai.shrink_dims(
        coreai.slice_(gf, [0, 0, 0, 1], [nb, out_h, out_w, 2], [1, 1, 1, 1]), [3]
    )
    col = denorm(gx, in_w)  # [nb, out_h, out_w]
    row = denorm(gy, in_h)

    hw = out_h * out_w
    chan = coreai.broadcast_to(
        coreai.reshape(coreai.constant(np.arange(c, dtype=np.float32)), [c, 1]),
        np.array([c, hw], dtype=np.int32),
    )
    chan_col = coreai.reshape(chan, [c * hw, 1])

    def sample_batch(n: int) -> Value:
        field = coreai.shrink_dims(
            coreai.slice_(xf, [n, 0, 0, 0], [n + 1, c, in_h, in_w], [1, 1, 1, 1]),
            [0],
        )
        if padding_mode == "zeros":
            # [b0,e0,b1,e1,b2,e2] for rank 3: 1px on H (dim 1) and W (dim 2).
            field = coreai.pad(
                field,
                np.array([0, 0, 1, 1, 1, 1], dtype=np.uint32),
                coreai.constant(0.0, dtype=np.float32),
                "constant",
            )

        def tile_over_channels(v: Value) -> Value:
            flat = coreai.reshape(v, [1, hw])
            rep = coreai.broadcast_to(flat, np.array([c, hw], dtype=np.int32))
            return coreai.reshape(rep, [c * hw, 1])

        row_n = coreai.shrink_dims(
            coreai.slice_(row, [n, 0, 0], [n + 1, out_h, out_w], [1, 1, 1]), [0]
        )
        col_n = coreai.shrink_dims(
            coreai.slice_(col, [n, 0, 0], [n + 1, out_h, out_w], [1, 1, 1]), [0]
        )
        coords = coreai.concat(
            1, [chan_col, tile_over_channels(row_n), tile_over_channels(col_n)]
        )
        sampled = coreai.interpolate_coordinates(field, coords, mode_attr)
        return coreai.expand_dims(coreai.reshape(sampled, [c, out_h, out_w]), [0])

    batches = [sample_batch(n) for n in range(nb)]
    out = batches[0] if nb == 1 else coreai.concat(0, batches)
    return out if f32 else coreai.cast(out, elem)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    "Gather": replace_gather,
    "GatherElements": replace_gather_elements,
    "GatherND": replace_gather_nd,
    "ScatterElements": replace_scatter_elements,
    "ScatterND": replace_scatter_nd,
    "OneHot": replace_one_hot,
    "NonZero": replace_non_zero,
    "EyeLike": replace_eye_like,
    "Compress": replace_compress,
    "GridSample": replace_grid_sample,
}
