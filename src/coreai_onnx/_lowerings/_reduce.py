# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for reduction, arg, scan, and sorting ONNX ops."""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, normalize_axis, operand, operands
from ._common import _INT32_MAX, _require_const, _variadic_reduce

# ---------------------------------------------------------------------------
# Reductions  (opset 18+ form: 'axes' as optional input 1)
#
# The coreai.reduce_* IR ops keep reduced dims as size 1 (the Python wrapper
# docstrings claim otherwise; the IR definition is authoritative), so
# keepdims=0 needs an explicit shrink_dims after the reduction.
# ---------------------------------------------------------------------------


def _reduce_axes(
    values_map: dict[str, Value], node: onnx.NodeProto, rank: int
) -> list[int] | None:
    """Resolved, normalized, sorted reduction axes — or None for an identity no-op."""
    node_attrs = attrs(node)
    raw = node_attrs.get("axes")  # opset <18 attribute form
    if raw is None:
        axes_v = operand(values_map, node, 1)
        if axes_v is not None:
            raw = _require_const(axes_v, node.op_type, "'axes' input").reshape(-1)
    if raw is None or len(raw) == 0:
        if node_attrs.get("noop_with_empty_axes", 0):
            return None
        return list(range(rank))
    normalized = sorted(normalize_axis(int(a), rank) for a in raw)
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"{node.op_type}: duplicate axes after normalization: {normalized}"
        )
    return normalized


def _square(x: Value) -> Value:
    return coreai.broadcasting_mul(x, x)


def _reduce(
    op_name: str,
    reduce_fn: Callable[..., Any],
    pre: Callable[[Value], Value] | None = None,
    post: Callable[[Value], Value] | None = None,
) -> Callable[..., Value]:
    """Lowering factory: result = post(reduce_fn(pre(x), axes)), then keepdims."""

    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        (x,) = operands(values_map, node, [0])
        axes = _reduce_axes(values_map, node, tensor_type(x).rank)
        if not axes:
            # None (noop_with_empty_axes) or [] (rank-0 input): reduce over
            # zero axes. Skip the reduction — coreai.reduce_* rejects an empty
            # axes tensor and shrink_dims([]) aborts the process — but keep
            # pre/post (e.g. ReduceL1 noop is |x|, not x).
            result = pre(x) if pre else x
            return post(result) if post else result
        result = reduce_fn(pre(x) if pre else x, axes)
        if post is not None:
            result = post(result)
        if not attrs(node).get("keepdims", 1):
            result = coreai.shrink_dims(result, axes)
        return result

    _lower.__name__ = f"replace_{op_name}"
    return _lower


def replace_reduce_log_sum_exp(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    """Stable logsumexp: m + log(sum(exp(x - m))) with m = max(x, axes)."""
    (x,) = operands(values_map, node, [0])
    axes = _reduce_axes(values_map, node, tensor_type(x).rank)
    if not axes:
        # Zero reduction axes (noop_with_empty_axes or rank-0 input):
        # log(sum(exp(x), axes=())) == x, and empty axes are invalid IR.
        return x
    # reduce_max keeps the reduced dims as size 1, so m broadcasts against x.
    m = coreai.reduce_max(x, axes)
    sum_exp = coreai.reduce_sum(coreai.exp(coreai.broadcasting_sub(x, m)), axes)
    result = coreai.broadcasting_add(m, coreai.log(sum_exp))
    if not attrs(node).get("keepdims", 1):
        result = coreai.shrink_dims(result, axes)
    return result


# ---------------------------------------------------------------------------
# ArgMax / ArgMin  (ties → smallest index)
# ---------------------------------------------------------------------------


def _arg_reduce(op_name: str, minimum: bool) -> Callable[..., Value]:
    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        (x,) = operands(values_map, node, [0])
        node_attrs = attrs(node)
        if node_attrs.get("select_last_index", 0):
            raise ValueError(f"{op_name}: select_last_index=1 is not supported")
        axis = normalize_axis(node_attrs.get("axis", 0), tensor_type(x).rank)
        if minimum:
            # Do not lower ArgMin as ArgMax(-x): signed integer minima overflow
            # under negation, and unsigned inputs are valid ONNX ArgMin inputs.
            order = coreai.argsort(x, axis, False, True)
            start = np.zeros(tensor_type(x).rank, dtype=np.int32)
            end = np.array(
                [1 if i == axis else _INT32_MAX for i in range(tensor_type(x).rank)],
                dtype=np.int32,
            )
            step = np.ones(tensor_type(x).rank, dtype=np.int32)
            result = coreai.cast(coreai.slice_(order, start, end, step), np.int32)
        else:
            # coreai.argmax keeps the reduced dim and resolves ties to the
            # smallest index, matching ONNX select_last_index=0 semantics.
            result = coreai.cast(coreai.argmax(x, axis), np.int32)
        if not node_attrs.get("keepdims", 1):
            result = coreai.shrink_dims(result, [axis])
        return result

    _lower.__name__ = f"replace_{op_name.lower()}"
    return _lower


# ---------------------------------------------------------------------------
# CumSum  (coreai.scan handles reverse natively; exclusive = shift via pad+slice)
# ---------------------------------------------------------------------------


def replace_cumsum(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, axis_v = operands(values_map, node, [0, 1])
    node_attrs = attrs(node)
    exclusive = node_attrs.get("exclusive", 0)
    reverse = node_attrs.get("reverse", 0)
    rank = tensor_type(x).rank
    axis = normalize_axis(
        int(_require_const(axis_v, "CumSum", "'axis' input").reshape(-1)[0]), rank
    )
    if exclusive:
        # exclusive cumsum = inclusive cumsum of x shifted one step along the
        # scan direction with a zero shifted in: pad one zero on the side the
        # scan starts from, then drop the element at the far end.
        dim = tensor_type(x).shape[axis]
        if dim < 0:
            raise ValueError("CumSum: exclusive=1 requires a static axis dimension")
        padding = [0] * (2 * rank)
        padding[2 * axis + (1 if reverse else 0)] = 1
        zero = coreai.constant(0, dtype=tensor_type(x).element_type)
        padded = coreai.pad(x, np.array(padding, dtype=np.uint32), zero, "constant")
        start = [0] * rank
        end = [_INT32_MAX] * rank
        start[axis], end[axis] = (1, dim + 1) if reverse else (0, dim)
        x = coreai.slice_(
            padded,
            np.array(start, dtype=np.int32),
            np.array(end, dtype=np.int32),
            np.array([1] * rank, dtype=np.int32),
        )
    return coreai.scan(x, np.uint32(axis), bool(reverse), combiner="sum")


# ---------------------------------------------------------------------------
# TopK  (sort + argsort + slice; outputs values and int32 indices)
# ---------------------------------------------------------------------------


def replace_topk(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> list[Value]:
    x, k_v = operands(values_map, node, [0, 1])
    node_attrs = attrs(node)
    rank = tensor_type(x).rank
    axis = normalize_axis(node_attrs.get("axis", -1), rank)
    descending = bool(node_attrs.get("largest", 1))
    k = int(_require_const(k_v, "TopK", "'k' input").reshape(-1)[0])
    dim = tensor_type(x).shape[axis]
    if dim >= 0 and k > dim:
        raise ValueError(f"TopK: k={k} exceeds axis {axis} dimension {dim}")
    # sorted=0 permits any order, so always producing sorted output is valid.
    indices = coreai.argsort(x, axis, descending, True)
    start = np.zeros(rank, dtype=np.int32)
    end = np.array(
        [k if i == axis else _INT32_MAX for i in range(rank)], dtype=np.int32
    )
    step = np.ones(rank, dtype=np.int32)
    indices = coreai.cast(coreai.slice_(indices, start, end, step), np.int32)
    # One sort, then gather the values — a second coreai.sort would re-sort
    # the whole tensor just to slice the top k.
    values = coreai.gather_along_axis(x, indices, axis)
    return [values, indices]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    # Variadic elementwise
    "Sum": _variadic_reduce(coreai.broadcasting_add),
    # Reductions
    "ReduceSum": _reduce("reduce_sum", coreai.reduce_sum),
    "ReduceMean": _reduce("reduce_mean", coreai.reduce_mean),
    "ReduceMax": _reduce("reduce_max", coreai.reduce_max),
    "ReduceMin": _reduce("reduce_min", coreai.reduce_min),
    "ReduceProd": _reduce("reduce_prod", coreai.reduce_product),
    "ReduceL1": _reduce("reduce_l1", coreai.reduce_sum, pre=coreai.abs_),
    "ReduceL2": _reduce("reduce_l2", coreai.reduce_sum, pre=_square, post=coreai.sqrt),
    "ReduceLogSum": _reduce("reduce_log_sum", coreai.reduce_sum, post=coreai.log),
    "ReduceLogSumExp": replace_reduce_log_sum_exp,
    "ReduceSumSquare": _reduce("reduce_sum_square", coreai.reduce_sum, pre=_square),
    # Arg ops / scans / sorting
    "ArgMax": _arg_reduce("ArgMax", minimum=False),
    "ArgMin": _arg_reduce("ArgMin", minimum=True),
    "CumSum": replace_cumsum,
    "TopK": replace_topk,
}
