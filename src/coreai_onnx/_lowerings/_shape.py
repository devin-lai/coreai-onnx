# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for shape and data-movement ONNX ops (Reshape ... Range)."""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import IntegerType, Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._type_mapping import coreai_type_from_onnx_dtype
from .._utils import attrs, normalize_axis, operand, operands
from ._common import _INT32_MAX, _const_array, _require_const

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def replace_identity(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    return x


# ---------------------------------------------------------------------------
# Cast / CastLike
# ---------------------------------------------------------------------------


def replace_cast(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    target = coreai_type_from_onnx_dtype(attrs(node)["to"])
    if tensor_type(x).element_type == target:
        return x
    return coreai.cast(x, target)


def replace_cast_like(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, target = operands(values_map, node, [0, 1])
    if tensor_type(x).element_type == tensor_type(target).element_type:
        return x
    return coreai.cast(x, tensor_type(target).element_type)


# ---------------------------------------------------------------------------
# Reshape  (shape input; 0 copies input dim unless allowzero, -1 inferred)
# ---------------------------------------------------------------------------


def replace_reshape(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, shape_v = operands(values_map, node, [0, 1])
    allowzero = attrs(node).get("allowzero", 0)
    arr = _const_array(shape_v)
    if arr is None:
        # Dynamic target shape: coreai.reshape infers static dims where it can.
        rank = tensor_type(x).rank
        n = tensor_type(shape_v).shape[0]
        if not allowzero and rank > 0 and n > 0:
            # ONNX (allowzero=0): a 0 in the runtime shape copies the input
            # dim. Only positions < rank can be copied; a 0 beyond the input
            # rank is invalid ONNX, so those positions pad with -1.
            in_shape = coreai.cast(coreai.get_shape(x), np.int32)
            if n > rank:
                in_shape = coreai.concat(
                    0, [in_shape, coreai.constant(np.full(n - rank, -1, np.int32))]
                )
            elif n < rank:
                in_shape = coreai.slice_(in_shape, [0], [n], [1])
            is_zero = coreai.broadcasting_equal(
                shape_v, coreai.constant(0, dtype=np.int32)
            )
            shape_v = coreai.broadcasting_where(is_zero, in_shape, shape_v)
        return coreai.reshape(x, shape_v)
    dims = [int(d) for d in arr.reshape(-1)]
    if allowzero and any(d == 0 for d in dims):
        raise ValueError(
            "Reshape: allowzero=1 with a 0-size output dimension is not supported"
        )
    if not allowzero:
        for i, d in enumerate(dims):
            if d == 0:
                in_dim = tensor_type(x).shape[i]
                if in_dim < 0:
                    raise ValueError(
                        "Reshape: target dim 0 copies a dynamic input dimension"
                    )
                dims[i] = in_dim
    # coreai.reshape resolves a single -1 (statically when the input shape is
    # known, otherwise via ReshapeOp's own inference).
    return coreai.reshape(x, dims)


# ---------------------------------------------------------------------------
# Transpose  (perm attr; default reverses the axes)
# ---------------------------------------------------------------------------


def replace_transpose(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    perm = attrs(node).get("perm")
    if perm is None:
        perm = list(reversed(range(rank)))
    perm = [normalize_axis(int(p), rank) for p in perm]
    return coreai.transpose(x, np.array(perm, dtype=np.uint32))


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

# Max inputs composed into one pad+where/pad+add chain; see replace_concat.
_CONCAT_CHUNK = 8


def _static_concat_with_pad_where(
    vals: list[Value], shapes: list[list[int]], axis: int
) -> Value:
    out_shape = shapes[0].copy()
    out_shape[axis] = sum(shape[axis] for shape in shapes)

    # Xcode's Performance runner initializes static tensor concat through
    # MPSGraph's GPU ConcatOpHandler, which aborts on common channel-concat
    # image models. Assemble static concat with pad+where instead.
    zero = coreai.constant(0, dtype=tensor_type(vals[0]).element_type)
    result: Value = coreai.broadcast_to(zero, np.array(out_shape, dtype=np.int32))
    true = coreai.constant(True, dtype=np.bool_)
    false = coreai.constant(False, dtype=np.bool_)
    offset = 0
    for val, shape in zip(vals, shapes, strict=True):
        if shape[axis] == 0:
            continue
        padding = np.zeros(2 * len(out_shape), dtype=np.uint32)
        padding[2 * axis] = offset
        padding[2 * axis + 1] = out_shape[axis] - offset - shape[axis]
        padded = coreai.pad(val, padding, zero)
        mask = coreai.pad(
            coreai.broadcast_to(true, np.array(shape, dtype=np.int32)),
            padding,
            false,
        )
        result = coreai.broadcasting_where(mask, padded, result)
        offset += shape[axis]
    return result


def _static_concat_with_pad_add(
    vals: list[Value], shapes: list[list[int]], axis: int
) -> Value:
    out_shape = shapes[0].copy()
    out_shape[axis] = sum(shape[axis] for shape in shapes)

    # Xcode's Performance runner initializes static tensor concat through
    # MPSGraph's GPU ConcatOpHandler, which aborts on common channel-concat
    # image models. Padding each input into its final non-overlapping slot and
    # adding those padded tensors avoids both native concat and bool masks.
    zero = coreai.constant(0, dtype=tensor_type(vals[0]).element_type)
    result: Value = coreai.broadcast_to(zero, np.array(out_shape, dtype=np.int32))
    offset = 0
    for val, shape in zip(vals, shapes, strict=True):
        if shape[axis] == 0:
            continue
        padding = np.zeros(2 * len(out_shape), dtype=np.uint32)
        padding[2 * axis] = offset
        padding[2 * axis + 1] = out_shape[axis] - offset - shape[axis]
        padded = coreai.pad(val, padding, zero)
        result = coreai.broadcasting_add(result, padded)
        offset += shape[axis]
    return result


def replace_concat(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    vals = [values_map[name] for name in node.input if name]
    if not vals:
        raise ValueError("Concat: expected at least one input")
    if len(vals) == 1:
        return vals[0]
    axis = normalize_axis(attrs(node)["axis"], tensor_type(vals[0]).rank)

    ranks = [tensor_type(v).rank for v in vals]
    if any(r != ranks[0] for r in ranks):
        raise ValueError(f"Concat: all inputs must have the same rank, got {ranks}")
    shapes = [list(tensor_type(v).shape) for v in vals]
    if any(dim < 0 for shape in shapes for dim in shape):
        return coreai.concat(axis, vals)
    elem_types = [tensor_type(v).element_type for v in vals]
    if any(elem_type != elem_types[0] for elem_type in elem_types[1:]):
        raise ValueError(
            "Concat: all inputs must have the same element type, got "
            f"{[str(elem_type) for elem_type in elem_types]}"
        )

    out_shape = shapes[0].copy()
    out_shape[axis] = sum(shape[axis] for shape in shapes)
    for i, shape in enumerate(shapes[1:], start=1):
        mismatched = [
            dim
            for dim, (got, want) in enumerate(zip(shape, out_shape, strict=True))
            if dim != axis and got != want
        ]
        if mismatched:
            raise ValueError(
                "Concat: input "
                f"{i} has incompatible shape {shape}; expected {out_shape} "
                f"outside axis {axis}"
            )

    elem_type = elem_types[0]
    if isinstance(elem_type, IntegerType) and elem_type.width != 1:
        compose = _static_concat_with_pad_add
    else:
        compose = _static_concat_with_pad_where

    # A single pad+where/pad+add chain over too many inputs becomes one fused
    # MPSGraph kernel whose argument table exceeds Metal's limit, aborting the
    # process at compile time ("Unable to get MPS kernel ndArrayIdentity ...
    # invalid location"); 12 inputs already crash the GPU and default units.
    # Compose wide concats as a tree of narrow ones instead.
    while len(vals) > _CONCAT_CHUNK:
        next_vals, next_shapes = [], []
        for i in range(0, len(vals), _CONCAT_CHUNK):
            chunk_vals = vals[i : i + _CONCAT_CHUNK]
            chunk_shapes = shapes[i : i + _CONCAT_CHUNK]
            if len(chunk_vals) == 1:
                next_vals.append(chunk_vals[0])
                next_shapes.append(chunk_shapes[0])
                continue
            chunk_out = chunk_shapes[0].copy()
            chunk_out[axis] = sum(shape[axis] for shape in chunk_shapes)
            next_vals.append(compose(chunk_vals, chunk_shapes, axis))
            next_shapes.append(chunk_out)
        vals, shapes = next_vals, next_shapes
    return compose(vals, shapes, axis)


# ---------------------------------------------------------------------------
# Split  (sizes via 'split' input, or equal chunks via num_outputs attr)
# ---------------------------------------------------------------------------


def replace_split(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> list[Value]:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    axis = normalize_axis(node_attrs.get("axis", 0), tensor_type(x).rank)
    split_v = operand(values_map, node, 1)
    if split_v is not None:
        sections = (
            _require_const(split_v, "Split", "'split' sizes")
            .reshape(-1)
            .astype(np.uint32)
        )
    else:
        num = node_attrs.get("num_outputs", len([o for o in node.output if o]))
        dim = tensor_type(x).shape[axis]
        if dim < 0:
            raise ValueError("Split: equal split requires a static split dimension")
        if num > dim:
            # ONNX Runtime rejects this ("Invalid num_outputs value of N");
            # the chunk formula below would underflow the uint32 sections.
            raise ValueError(
                f"Split: num_outputs={num} exceeds the split axis size ({dim})"
            )
        chunk = -(-dim // num)  # ceil; last chunk gets the remainder (opset 18+)
        sections = np.array([chunk] * (num - 1) + [dim - chunk * (num - 1)], np.uint32)
    results = coreai.split(x, sections, np.int32(axis))
    if isinstance(results, Value):
        return [results]  # single-output split folds to identity
    return list(results)


# ---------------------------------------------------------------------------
# Slice  (starts/ends/axes/steps inputs; coreai.slice_ supports negative steps)
# ---------------------------------------------------------------------------


def _resolve_onnx_slice_bounds(
    start: int, end: int, step: int, dim: int
) -> tuple[int, int]:
    """Resolve ONNX Slice start/end (incl. negatives and INT32 sentinels)."""
    if start < 0:
        start += dim
    if end < 0:
        end += dim
    if step > 0:
        start = max(0, min(start, dim))
        end = max(0, min(end, dim))
    else:
        start = max(0, min(start, dim - 1))
        end = max(-1, min(end, dim - 1))
    return start, end


def _runtime_slice_bound(v: Value, dim: Value) -> Value:
    """Resolve a positive-step ONNX bound at runtime: add dim to negatives,
    then clamp to [0, dim] (INT32_MAX sentinels clamp down to dim)."""
    zero = coreai.constant(0, dtype=np.int32)
    wrapped = coreai.broadcasting_add(v, dim)
    v = coreai.broadcasting_where(coreai.broadcasting_greater(zero, v), wrapped, v)
    return coreai.broadcasting_minimum(coreai.broadcasting_maximum(v, zero), dim)


def replace_slice(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    starts_v, ends_v = operands(values_map, node, [1, 2])
    axes_v = operand(values_map, node, 3)
    steps_v = operand(values_map, node, 4)

    starts_arr = _const_array(starts_v)
    ends_arr = _const_array(ends_v)
    starts = None if starts_arr is None else [int(s) for s in starts_arr.reshape(-1)]
    ends = None if ends_arr is None else [int(e) for e in ends_arr.reshape(-1)]
    n = len(starts) if starts is not None else tensor_type(starts_v).shape[0]
    if n < 0:
        raise ValueError("Slice: runtime 'starts' must have a static length")
    if axes_v is not None:
        axes_arr = _require_const(axes_v, "Slice", "'axes' input").reshape(-1)
        axes = [normalize_axis(int(a), rank) for a in axes_arr]
    else:
        axes = list(range(n))
    if steps_v is not None:
        steps = [
            int(s)
            for s in _require_const(steps_v, "Slice", "'steps' input").reshape(-1)
        ]
    else:
        steps = [1] * len(axes)

    # Per-axis entries are ints where resolvable at compile time, otherwise
    # runtime int32 [1] Values (the ONNX starts/ends inputs narrow to si32).
    start_full: list[Any] = [0] * rank
    end_full: list[Any] = [_INT32_MAX] * rank
    step_full = [1] * rank
    reversed_axes: list[int] = []
    shp: Value | None = None

    def dim_value(a: int) -> Value:
        nonlocal shp
        if shp is None:
            shp = coreai.cast(coreai.get_shape(x), np.int32)
        return coreai.slice_(shp, [a], [a + 1], [1])

    for i, (a, st) in enumerate(zip(axes, steps, strict=True)):
        s = starts[i] if starts is not None else None
        e = ends[i] if ends is not None else None
        dim = tensor_type(x).shape[a]
        if st < 0:
            # The runtime mishandles end == -1 with a negative stride (it wraps
            # the -1 to dim-1, yielding an empty result), so decompose every
            # negative-step slice as reverse + positive-step slice instead.
            # Index i on the original axis is dim-1-i on the reversed axis.
            if s is None or e is None:
                raise ValueError(
                    "Slice: negative steps require compile-time 'starts'/'ends'"
                )
            if dim < 0:
                raise ValueError(
                    "Slice: negative steps require a static dimension size"
                )
            s, e = _resolve_onnx_slice_bounds(s, e, st, dim)
            reversed_axes.append(a)
            start_full[a], end_full[a], step_full[a] = dim - 1 - s, dim - 1 - e, -st
        elif s is not None and e is not None and dim >= 0:
            start_full[a], end_full[a] = _resolve_onnx_slice_bounds(s, e, st, dim)
            step_full[a] = st
        elif s is not None and e is not None and s >= 0 and e >= 0:
            # Dynamic dim with non-negative const bounds: SliceOp clamps
            # over-large ends itself.
            start_full[a], end_full[a], step_full[a] = s, e, st
        else:
            # Runtime starts/ends, or negative const bounds on a dynamic axis
            # (e.g. the last-token slice starts=[-1]): resolve per ONNX rules
            # with the dim read at runtime.
            d = dim_value(a)
            sv = (
                coreai.constant(np.array([s], dtype=np.int32))
                if s is not None
                else coreai.slice_(starts_v, [i], [i + 1], [1])
            )
            ev = (
                coreai.constant(np.array([e], dtype=np.int32))
                if e is not None
                else coreai.slice_(ends_v, [i], [i + 1], [1])
            )
            start_full[a] = _runtime_slice_bound(sv, d)
            end_full[a] = _runtime_slice_bound(ev, d)
            step_full[a] = st

    if reversed_axes:
        x = coreai.reverse(x, np.array(reversed_axes, dtype=np.int32))
    if not any(isinstance(v, Value) for v in start_full + end_full):
        return coreai.slice_(
            x,
            np.array(start_full, dtype=np.int32),
            np.array(end_full, dtype=np.int32),
            np.array(step_full, dtype=np.int32),
        )

    def as_vector(items: list[Any]) -> Value:
        parts = [
            v if isinstance(v, Value) else coreai.constant(np.array([v], np.int32))
            for v in items
        ]
        return coreai.concat(0, parts)

    return coreai.slice_(
        x,
        as_vector(start_full),
        as_vector(end_full),
        np.array(step_full, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Squeeze / Unsqueeze  (axes as input 1, opset 13+ form)
# ---------------------------------------------------------------------------


def replace_squeeze(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    axes_v = operand(values_map, node, 1)
    if axes_v is not None:
        arr = _require_const(axes_v, "Squeeze", "'axes' input")
        axes = sorted(normalize_axis(int(a), rank) for a in arr.reshape(-1))
    else:
        shape = tensor_type(x).shape
        if any(d < 0 for d in shape):
            raise ValueError(
                "Squeeze without 'axes' requires a fully static input shape"
            )
        axes = [i for i, d in enumerate(shape) if d == 1]
    if not axes:
        return x
    return coreai.shrink_dims(x, axes)


def replace_unsqueeze(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, axes_v = operands(values_map, node, [0, 1])
    arr = _require_const(axes_v, "Unsqueeze", "'axes' input")
    out_rank = tensor_type(x).rank + arr.size
    axes = sorted(normalize_axis(int(a), out_rank) for a in arr.reshape(-1))
    return coreai.expand_dims(x, axes)


# ---------------------------------------------------------------------------
# Flatten  (reshape to [prod(:axis), prod(axis:)])
# ---------------------------------------------------------------------------


def replace_flatten(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    axis = normalize_axis(attrs(node).get("axis", 1), rank)  # axis == rank is valid
    shape = list(tensor_type(x).shape)
    if all(d >= 0 for d in shape[:axis]):
        return coreai.reshape(x, [math.prod(shape[:axis]), -1])
    if all(d >= 0 for d in shape[axis:]):
        return coreai.reshape(x, [-1, math.prod(shape[axis:])])
    # Both halves contain dynamic dims: compute the head extent at runtime.
    shp = coreai.cast(coreai.get_shape(x), np.int32)
    head = coreai.reduce_product(coreai.slice_(shp, [0], [axis], [1]), [0])
    target = coreai.concat(
        0, [coreai.expand_dims(head, [0]), coreai.constant([-1], dtype=np.int32)]
    )
    return coreai.reshape(x, target)


# ---------------------------------------------------------------------------
# Expand  (numpy-style bidirectional broadcast; a 1 in shape keeps input dim)
# ---------------------------------------------------------------------------


def replace_expand(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, shape_v = operands(values_map, node, [0, 1])
    arr = _const_array(shape_v)
    in_shape = tuple(tensor_type(x).shape)
    if arr is not None and all(d >= 0 for d in in_shape):
        # coreai.broadcast_to requires non-1 target dims to match exactly, so
        # resolve ONNX's bidirectional broadcast to the concrete output shape.
        target = np.broadcast_shapes(in_shape, tuple(int(d) for d in arr.reshape(-1)))
        return coreai.broadcast_to(x, np.array(target, dtype=np.uint32))
    # Dynamic input dims and/or runtime shape: resolve the bidirectional
    # broadcast at runtime (a 1 in `shape` keeps the input dim and `shape` may
    # have lower rank than the input — broadcast_to alone mishandles both).
    # The ui32 cast also keeps the broadcast_to builder from introspecting a
    # signed shape's owner, which crashes on block arguments.
    target_v: Value | np.ndarray
    if arr is not None:
        target_v = arr.reshape(-1).astype(np.uint32)
    else:
        target_v = coreai.cast(shape_v, np.uint32)
    return coreai.broadcast_to(
        x, coreai.broadcast_shapes(coreai.get_shape(x), target_v)
    )


# ---------------------------------------------------------------------------
# Tile  (repeats input; ONNX requires len(repeats) == rank)
# ---------------------------------------------------------------------------


def replace_tile(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, repeats_v = operands(values_map, node, [0, 1])
    arr = _const_array(repeats_v)
    if arr is None:
        return coreai.tile(x, coreai.cast(repeats_v, np.uint32))
    return coreai.tile(x, arr.reshape(-1).astype(np.uint32))


# ---------------------------------------------------------------------------
# Pad  (pads input [x1_begin, x2_begin, ..., x1_end, x2_end, ...])
#
# ONNX modes map onto Core AI's: edge -> replicate, wrap -> circular. coreai
# also offers 'symmetric' (edge-inclusive reflection), which ONNX Pad has no
# equivalent for; unmapped modes raise with the mode named.
# ---------------------------------------------------------------------------

_ONNX_TO_COREAI_PAD_MODE = {
    "constant": "constant",
    "reflect": "reflect",
    "edge": "replicate",
    "wrap": "circular",
}


def replace_pad(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, pads_v = operands(values_map, node, [0, 1])
    mode = attrs(node).get("mode", "constant")
    coreai_mode = _ONNX_TO_COREAI_PAD_MODE.get(mode)
    if coreai_mode is None:
        raise ValueError(
            f"Pad: mode '{mode}' is not supported "
            "(supported: constant, reflect, edge, wrap)"
        )
    pads = _require_const(pads_v, "Pad", "'pads' input").reshape(-1)
    rank = tensor_type(x).rank
    axes_v = operand(values_map, node, 3)
    if axes_v is not None:
        axes = [
            normalize_axis(int(a), rank)
            for a in _require_const(axes_v, "Pad", "'axes' input").reshape(-1)
        ]
    else:
        axes = list(range(rank))
    if len(axes) != len(set(axes)):
        raise ValueError("Pad: duplicate axes")

    n = len(axes)
    # Reorder ONNX [b1, b2, ..., e1, e2, ...] to coreai [b1, e1, b2, e2, ...].
    padding = [0] * (2 * rank)
    for i, a in enumerate(axes):
        padding[2 * a] = int(pads[i])
        padding[2 * a + 1] = int(pads[n + i])
    if any(p < 0 for p in padding):
        raise ValueError("Pad: negative pads (cropping) are not supported")
    if not any(padding):
        return x

    value_v = operand(values_map, node, 2)
    value: Value
    if value_v is None:
        value = coreai.constant(0, dtype=tensor_type(x).element_type)
    elif tensor_type(value_v).element_type != tensor_type(x).element_type:
        value = coreai.cast(value_v, tensor_type(x).element_type)
    else:
        value = value_v
    return coreai.pad(x, np.array(padding, dtype=np.uint32), value, coreai_mode)


# ---------------------------------------------------------------------------
# Shape / Size
# ---------------------------------------------------------------------------


def replace_shape(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    node_attrs = attrs(node)
    start = node_attrs.get("start", 0)
    end = node_attrs.get("end", rank)
    start = max(0, min(rank, normalize_axis(start, rank)))
    end = max(0, min(rank, normalize_axis(end, rank)))
    shape = list(tensor_type(x).shape)
    if all(d >= 0 for d in shape[start:end]):
        return coreai.constant(np.array(shape[start:end], dtype=np.int32))
    shp = coreai.cast(coreai.get_shape(x), np.int32)
    if (start, end) == (0, rank):
        return shp
    return coreai.slice_(shp, [start], [end], [1])


def replace_size(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    shape = list(tensor_type(x).shape)
    if all(d >= 0 for d in shape):
        return coreai.constant(np.array(math.prod(shape), dtype=np.int32))
    prod = coreai.reduce_product(coreai.cast(coreai.get_shape(x), np.int32), [0])
    # reduce_product keeps a rank-1 result; ONNX Size is a scalar.
    return coreai.reshape(prod, [])


# ---------------------------------------------------------------------------
# DepthToSpace / SpaceToDepth  (reshape + transpose + reshape, ONNX formulas)
# ---------------------------------------------------------------------------


def _static_nchw(x: Value, op_name: str) -> tuple[int, int, int, int]:
    if tensor_type(x).rank != 4:
        raise ValueError(f"{op_name}: input must be rank 4 (NCHW)")
    n, c, h, w = tensor_type(x).shape
    if min(n, c, h, w) < 0:
        raise ValueError(f"{op_name}: requires a fully static input shape")
    return n, c, h, w


def replace_depth_to_space(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    b = node_attrs["blocksize"]
    mode = node_attrs.get("mode", "DCR")
    n, c, h, w = _static_nchw(x, "DepthToSpace")
    if c % (b * b) != 0:
        raise ValueError(
            f"DepthToSpace: channel count C={c} is not divisible by blocksize²={b * b}"
        )
    t: Value
    if mode == "DCR":
        t = coreai.reshape(x, [n, b, b, c // (b * b), h, w])
        t = coreai.transpose(t, np.array([0, 3, 4, 1, 5, 2], dtype=np.uint32))
    elif mode == "CRD":
        t = coreai.reshape(x, [n, c // (b * b), b, b, h, w])
        t = coreai.transpose(t, np.array([0, 1, 4, 2, 5, 3], dtype=np.uint32))
    else:
        raise ValueError(f"DepthToSpace: unknown mode '{mode}'")
    return coreai.reshape(t, [n, c // (b * b), h * b, w * b])


def replace_space_to_depth(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    b = attrs(node)["blocksize"]
    n, c, h, w = _static_nchw(x, "SpaceToDepth")
    if h % b != 0 or w % b != 0:
        raise ValueError(
            f"SpaceToDepth: H={h} or W={w} is not divisible by blocksize={b}"
        )
    t: Value = coreai.reshape(x, [n, c, h // b, b, w // b, b])
    t = coreai.transpose(t, np.array([0, 3, 5, 1, 2, 4], dtype=np.uint32))
    return coreai.reshape(t, [n, c * b * b, h // b, w // b])


# ---------------------------------------------------------------------------
# Trilu  (constant keep-mask + where; static trailing dims)
# ---------------------------------------------------------------------------


def replace_trilu(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    upper = attrs(node).get("upper", 1)
    k = 0
    k_v = operand(values_map, node, 1)
    if k_v is not None:
        k = int(_require_const(k_v, "Trilu", "'k' input").reshape(-1)[0])
    rows, cols = tensor_type(x).shape[-2], tensor_type(x).shape[-1]
    if rows < 0 or cols < 0:
        raise ValueError("Trilu: requires static trailing (row, col) dimensions")
    # Build the keep-mask from two O(rows+cols) iota constants instead of a
    # dense (rows, cols) matrix, which bloats the model for large causal masks.
    # Clamping k keeps the int32 shift arithmetic below overflow-free without
    # changing which elements are kept.
    k = max(-(rows + 1), min(k, cols + 1))
    row_i = coreai.constant(np.arange(rows, dtype=np.int32).reshape(rows, 1))
    col_i = coreai.constant(np.arange(cols, dtype=np.int32))
    if upper:
        # keep j >= i + k  <=>  j > i + (k - 1)
        keep = coreai.broadcasting_greater(
            col_i,
            coreai.broadcasting_add(row_i, coreai.constant(k - 1, dtype=np.int32)),
        )
    else:
        # keep j <= i + k  <=>  i + (k + 1) > j
        keep = coreai.broadcasting_greater(
            coreai.broadcasting_add(row_i, coreai.constant(k + 1, dtype=np.int32)),
            col_i,
        )
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    return coreai.broadcasting_where(keep, x, zero)


# ---------------------------------------------------------------------------
# Constant / ConstantOfShape
#
# preprocess folds top-level Constant nodes; this lowering covers any that
# survive (e.g. inside future subgraph support).
# ---------------------------------------------------------------------------


def replace_constant(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    node_attrs = attrs(node)
    if "value" in node_attrs:
        return coreai.constant(node_attrs["value"])
    if "value_float" in node_attrs:
        return coreai.constant(np.array(node_attrs["value_float"], dtype=np.float32))
    if "value_int" in node_attrs:
        return coreai.constant(np.array(node_attrs["value_int"], dtype=np.int32))
    if "value_floats" in node_attrs:
        return coreai.constant(np.array(node_attrs["value_floats"], dtype=np.float32))
    if "value_ints" in node_attrs:
        return coreai.constant(np.array(node_attrs["value_ints"], dtype=np.int32))
    raise ValueError(
        "Constant: only value/value_float/value_int/value_floats/value_ints "
        "attributes are supported"
    )


def replace_constant_of_shape(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (shape_v,) = operands(values_map, node, [0])
    value = attrs(node).get("value")
    if value is None:
        value = np.zeros(1, dtype=np.float32)
    fill = value.reshape(-1)[0]
    arr = _const_array(shape_v)
    if arr is not None:
        dims = [int(d) for d in arr.reshape(-1)]
        return coreai.constant(np.full(dims, fill))
    # ui32 cast: the broadcast_to builder introspects a signed shape's owner,
    # which crashes when the shape is a block argument (graph input).
    return coreai.broadcast_to(
        coreai.constant(np.array(fill)), coreai.cast(shape_v, np.uint32)
    )


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


def replace_range(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    start_v, limit_v, delta_v = operands(values_map, node, [0, 1, 2])
    start_a = _const_array(start_v)
    limit_a = _const_array(limit_v)
    delta_a = _const_array(delta_v)
    if start_a is not None and limit_a is not None and delta_a is not None:
        start, limit, delta = (a.reshape(-1)[0] for a in (start_a, limit_a, delta_a))
        return coreai.constant(np.arange(start, limit, delta))

    def to_scalar(v: Value) -> Value:
        if tensor_type(v).rank > 0:
            return coreai.shrink_dims(v, list(range(tensor_type(v).rank)))
        return v

    return coreai.range_(to_scalar(start_v), to_scalar(limit_v), to_scalar(delta_v))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    "Reshape": replace_reshape,
    "Transpose": replace_transpose,
    "Concat": replace_concat,
    "Split": replace_split,
    "Slice": replace_slice,
    "Squeeze": replace_squeeze,
    "Unsqueeze": replace_unsqueeze,
    "Flatten": replace_flatten,
    "Expand": replace_expand,
    "Tile": replace_tile,
    "Pad": replace_pad,
    "Identity": replace_identity,
    "Cast": replace_cast,
    "CastLike": replace_cast_like,
    "Shape": replace_shape,
    "Size": replace_size,
    "DepthToSpace": replace_depth_to_space,
    "SpaceToDepth": replace_space_to_depth,
    "Trilu": replace_trilu,
    "Constant": replace_constant,
    "ConstantOfShape": replace_constant_of_shape,
    "Range": replace_range,
}
