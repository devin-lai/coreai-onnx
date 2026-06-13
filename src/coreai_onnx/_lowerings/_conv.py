# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for convolution, pooling, and LRN ONNX ops.

All ops here operate on NCHW-style layouts with 1-3 spatial dims. The Core AI
conv/pool primitives are 2d/3d only and take no padding (except
conv_transpose), so 1d is expanded to 2d with a unit H dim and padding is
emitted as an explicit ``coreai.pad`` first.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import FloatType, Location, RankedTensorType, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, operand, operands
from ._common import _INT32_MAX

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _int_list(raw: Any, default: list[int]) -> list[int]:
    return default if raw is None else [int(v) for v in raw]


def _static_spatial(x: Value, op_name: str) -> list[int]:
    sizes = list(tensor_type(x).shape[2:])
    if any(d < 0 for d in sizes):
        raise ValueError(f"{op_name}: requires static spatial input dims, got {sizes}")
    return sizes


def _resolve_pads(
    op_name: str,
    node_attrs: dict[str, Any],
    in_sizes: list[int],
    kernel: list[int],
    strides: list[int],
    dilations: list[int],
) -> tuple[list[int], list[int]]:
    """Per-spatial-dim (begin, end) pads from the pads/auto_pad attributes."""
    s = len(kernel)
    auto_pad = node_attrs.get("auto_pad", "NOTSET")
    if auto_pad in ("", "NOTSET"):
        pads = _int_list(node_attrs.get("pads"), [0] * (2 * s))
        return pads[:s], pads[s:]
    if auto_pad == "VALID":
        return [0] * s, [0] * s
    if auto_pad not in ("SAME_UPPER", "SAME_LOWER"):
        raise ValueError(f"{op_name}: unsupported auto_pad '{auto_pad}'")
    begins, ends = [], []
    for i in range(s):
        if in_sizes[i] < 0:
            raise ValueError(f"{op_name}: auto_pad '{auto_pad}' needs static dims")
        out = -(-in_sizes[i] // strides[i])  # ceil(in / stride)
        total = max(
            0, (out - 1) * strides[i] + (kernel[i] - 1) * dilations[i] + 1 - in_sizes[i]
        )
        half = total // 2
        if auto_pad == "SAME_UPPER":
            begins.append(half)
            ends.append(total - half)
        else:
            begins.append(total - half)
            ends.append(half)
    return begins, ends


def _pad_spatial(x: Value, begins: list[int], ends: list[int], value: Value) -> Value:
    """Pad the spatial dims of an NC... tensor (no-op when all pads are 0)."""
    if not any(begins) and not any(ends):
        return x
    padding = [0, 0, 0, 0]
    for b, e in zip(begins, ends, strict=True):
        padding += [b, e]
    return coreai.pad(x, np.array(padding, dtype=np.uint32), value)


def _add_channel_bias(result: Value, bias: Value, spatial: int) -> Value:
    return coreai.broadcasting_add(
        result, coreai.reshape(bias, [1, tensor_type(bias).shape[0]] + [1] * spatial)
    )


def _pool_out_sizes(
    padded: list[int], kernel: list[int], strides: list[int], dilations: list[int]
) -> list[int]:
    return [
        (p - d * (k - 1) - 1) // s + 1
        for p, k, s, d in zip(padded, kernel, strides, dilations, strict=True)
    ]


def _ceil_mode_ends(
    in_sizes: list[int],
    kernel: list[int],
    strides: list[int],
    dilations: list[int],
    begins: list[int],
    ends: list[int],
) -> list[int]:
    """End pads that make floor-mode pooling match ONNX ceil_mode=1 output.

    ONNX: out = ceil((in + pads - dilation*(kernel-1) - 1) / stride) + 1, but
    a window must start inside the input or begin padding — windows starting
    in the end padding are dropped (MaxPool spec note / onnx op_pool_common).
    Those `out` windows only touch taps < (out-1)*stride + window extent, so
    padding to exactly that extent makes _pool_out_sizes yield `out` (this may
    be more or less end padding than requested; the difference is never read).
    """
    new_ends = []
    for n, k, s, d, b, e in zip(
        in_sizes, kernel, strides, dilations, begins, ends, strict=True
    ):
        ext = d * (k - 1) + 1
        out = -((n + b + e - ext) // -s) + 1
        if (out - 1) * s >= n + b:
            out -= 1
        new_ends.append(max(0, (out - 1) * s + ext - n - b))
    return new_ends


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------


def _conv_values(
    x: Value,
    weight: Value,
    bias: Value | None,
    node_attrs: dict[str, Any],
    op_name: str = "Conv",
) -> Value:
    rank = tensor_type(x).rank
    if rank not in (3, 4, 5):
        raise ValueError(
            f"{op_name}: only 1d/2d/3d convolutions are supported, rank={rank}"
        )
    s = rank - 2
    strides = _int_list(node_attrs.get("strides"), [1] * s)
    dilations = _int_list(node_attrs.get("dilations"), [1] * s)
    group = int(node_attrs.get("group", 1))
    kernel = _int_list(
        node_attrs.get("kernel_shape"), list(tensor_type(weight).shape[2:])
    )

    begins, ends = _resolve_pads(
        op_name, node_attrs, list(tensor_type(x).shape[2:]), kernel, strides, dilations
    )
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    if s == 1:
        # Conv1d: insert a unit H dim, conv2d, drop it again.
        x = _pad_spatial(coreai.expand_dims(x, [2]), [0, *begins], [0, *ends], zero)
        weight = coreai.expand_dims(weight, [2])
        result = coreai.shrink_dims(
            coreai.conv2d(x, weight, [1, *strides], [1, *dilations], group), [2]
        )
    elif s == 2:
        x = _pad_spatial(x, begins, ends, zero)
        result = coreai.conv2d(x, weight, strides, dilations, group)
    else:
        x = _pad_spatial(x, begins, ends, zero)
        result = coreai.conv3d(
            x,
            weight,
            coreai.constant(strides, np.uint32),
            coreai.constant(dilations, np.uint32),
            coreai.constant(group, np.uint32),
        )
    if bias is not None:
        result = _add_channel_bias(result, bias, s)
    return result


def replace_conv(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, weight = operands(values_map, node, [0, 1])
    bias = operand(values_map, node, 2)
    return _conv_values(x, weight, bias, attrs(node))


# ---------------------------------------------------------------------------
# ConvTranspose
#
# The coreai conv_transpose primitives use symmetric (torch-style) padding, so
# the ONNX form is lowered as: run with padding=0/output_pad=0 (the "full"
# output), then crop pads from each side. output_padding crops less at the
# end — or appends zeros where the pad to crop is smaller than output_padding.
# ---------------------------------------------------------------------------


def replace_conv_transpose(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, weight = operands(values_map, node, [0, 1])
    bias = operand(values_map, node, 2)
    rank = tensor_type(x).rank
    if rank not in (3, 4, 5):
        raise ValueError(
            f"ConvTranspose: only 1d/2d/3d are supported, input rank={rank}"
        )
    s = rank - 2
    node_attrs = attrs(node)
    strides = _int_list(node_attrs.get("strides"), [1] * s)
    dilations = _int_list(node_attrs.get("dilations"), [1] * s)
    group = int(node_attrs.get("group", 1))
    kernel = _int_list(
        node_attrs.get("kernel_shape"), list(tensor_type(weight).shape[2:])
    )
    output_padding = _int_list(node_attrs.get("output_padding"), [0] * s)

    in_sizes = _static_spatial(x, "ConvTranspose")
    # Output sizes of the primitive with padding=0 and output_pad=0.
    full = [
        strides[i] * (in_sizes[i] - 1) + (kernel[i] - 1) * dilations[i] + 1
        for i in range(s)
    ]

    output_shape = node_attrs.get("output_shape")
    auto_pad = node_attrs.get("auto_pad", "NOTSET")
    if output_shape is not None or auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        # ONNX: total_padding[i] = full[i] + output_padding[i] - target[i],
        # split per auto_pad (extra pad goes to the start unless SAME_UPPER).
        target = (
            [int(v) for v in output_shape]
            if output_shape is not None
            else [in_sizes[i] * strides[i] for i in range(s)]
        )
        begins, ends = [], []
        for i in range(s):
            total = full[i] + output_padding[i] - target[i]
            if total < 0:
                raise ValueError(
                    f"ConvTranspose: output_shape {target} exceeds the maximum"
                    " computable output size "
                    f"{[f + o for f, o in zip(full, output_padding, strict=True)]}"
                )
            half = total // 2
            if auto_pad == "SAME_UPPER":
                begins.append(half)
                ends.append(total - half)
            else:
                begins.append(total - half)
                ends.append(half)
    elif auto_pad == "VALID":
        # auto_pad != NOTSET takes precedence over any (spec-invalid) pads.
        begins, ends = [0] * s, [0] * s
    elif auto_pad not in ("", "NOTSET"):
        raise ValueError(f"ConvTranspose: unsupported auto_pad '{auto_pad}'")
    else:
        pads = _int_list(node_attrs.get("pads"), [0] * (2 * s))
        begins, ends = pads[:s], pads[s:]

    is_1d = s == 1
    if is_1d:
        # ConvTranspose1d: insert a unit H dim, run the 2d primitive with
        # unit stride/dilation/crop there, and drop the dim again at the end.
        x = coreai.expand_dims(x, [2])
        weight = coreai.expand_dims(weight, [2])
        strides, dilations = [1, *strides], [1, *dilations]
        output_padding = [0, *output_padding]
        begins, ends, full = [0, *begins], [0, *ends], [1, *full]
        rank = 4

    conv_transpose = coreai.conv_transpose2d if rank == 4 else coreai.conv_transpose3d
    result = conv_transpose(
        input=x,
        weight=weight,
        stride=coreai.constant(strides, np.uint32),
        padding=coreai.constant([0] * len(strides), np.uint32),
        dilation=coreai.constant(dilations, np.uint32),
        output_pad=coreai.constant([0] * len(strides), np.uint32),
        groups=coreai.constant(group, np.uint32),
    )

    end_crops = [max(e - o, 0) for e, o in zip(ends, output_padding, strict=True)]
    end_pads = [max(o - e, 0) for e, o in zip(ends, output_padding, strict=True)]
    if any(begins) or any(end_crops):
        result = coreai.slice_(
            result,
            [0, 0, *begins],
            [_INT32_MAX, _INT32_MAX]
            + [f - c for f, c in zip(full, end_crops, strict=True)],
            [1] * rank,
        )
    if any(end_pads):
        padding = [0, 0, 0, 0]
        for p in end_pads:
            padding += [0, p]
        result = coreai.pad(
            result,
            np.array(padding, dtype=np.uint32),
            coreai.constant(0, dtype=tensor_type(x).element_type),
        )
    if is_1d:
        result = coreai.shrink_dims(result, [2])
    if bias is not None:
        result = _add_channel_bias(result, bias, s)
    return result


# ---------------------------------------------------------------------------
# MaxPool / AveragePool
# ---------------------------------------------------------------------------


def _pool_setup(
    op_name: str, values_map: dict[str, Value], node: onnx.NodeProto
) -> tuple[Value, int, dict[str, Any], list[int], list[int], list[int]]:
    """Common attribute handling. Returns (x, spatial_rank, attrs, kernel,
    strides, dilations)."""
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    if rank not in (3, 4, 5):
        raise ValueError(f"{op_name}: only 1d/2d/3d pooling is supported, rank={rank}")
    node_attrs = attrs(node)
    s = rank - 2
    kernel = _int_list(node_attrs.get("kernel_shape"), [])
    if len(kernel) != s:
        raise ValueError(f"{op_name}: kernel_shape must have {s} values, got {kernel}")
    strides = _int_list(node_attrs.get("strides"), [1] * s)
    dilations = _int_list(node_attrs.get("dilations"), [1] * s)
    return x, s, node_attrs, kernel, strides, dilations


def replace_max_pool(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    if len(node.output) > 1 and node.output[1]:
        raise ValueError("MaxPool: the optional Indices output is not supported")
    x, s, node_attrs, kernel, strides, dilations = _pool_setup(
        "MaxPool", values_map, node
    )
    if node_attrs.get("storage_order", 0):
        raise ValueError("MaxPool: storage_order=1 is not supported")
    if not isinstance(tensor_type(x).element_type, FloatType):
        # ONNX MaxPool-12+ allows int8/uint8, but the Core AI runtime cannot
        # load integer pooling programs (Program load failure 0x10004).
        raise ValueError(
            "MaxPool: non-float input dtypes are not supported by the Core AI "
            f"runtime, got element type {tensor_type(x).element_type}"
        )
    in_sizes = _static_spatial(x, "MaxPool")
    begins, ends = _resolve_pads(
        "MaxPool", node_attrs, in_sizes, kernel, strides, dilations
    )
    if node_attrs.get("ceil_mode", 0):
        ends = _ceil_mode_ends(in_sizes, kernel, strides, dilations, begins, ends)
    # coreai.maxpool* take no padding: pad with -inf so padding never wins.
    neg_inf = coreai.cast(np.float32("-inf"), tensor_type(x).element_type)
    is_1d = s == 1
    if is_1d:
        x = coreai.expand_dims(x, [2])
        in_sizes = [1, *in_sizes]
        kernel, strides, dilations = [1, *kernel], [1, *strides], [1, *dilations]
        begins, ends = [0, *begins], [0, *ends]
    x = _pad_spatial(x, begins, ends, neg_inf)
    padded = [sz + b + e for sz, b, e in zip(in_sizes, begins, ends, strict=True)]
    out_sizes = _pool_out_sizes(padded, kernel, strides, dilations)
    n, c = tensor_type(x).shape[0], tensor_type(x).shape[1]
    maxpool = coreai.maxpool2d if len(kernel) == 2 else coreai.maxpool3d
    result = maxpool(
        output=RankedTensorType.get([n, c, *out_sizes], tensor_type(x).element_type),
        input=x,
        kernel_size=np.array(kernel, np.uint32),
        stride=np.array(strides, np.uint32),
        dilation=np.array(dilations, np.uint32),
        ceil_mode=False,
    )
    return coreai.shrink_dims(result, [2]) if is_1d else result


def replace_average_pool(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, s, node_attrs, kernel, strides, dilations = _pool_setup(
        "AveragePool", values_map, node
    )
    count_include_pad = int(node_attrs.get("count_include_pad", 0))
    in_sizes = _static_spatial(x, "AveragePool")
    begins, ends = _resolve_pads(
        "AveragePool", node_attrs, in_sizes, kernel, strides, dilations
    )
    orig_ends = ends
    if node_attrs.get("ceil_mode", 0):
        ends = _ceil_mode_ends(in_sizes, kernel, strides, dilations, begins, ends)

    # Lower as zero-pad + sumpool, then divide by the per-window element count,
    # a compile-time constant tensor (separable product of per-dim tap counts).
    # count_include_pad=0 counts only original-input taps; count_include_pad=1
    # also counts explicit pad taps, but never the implicit ceil_mode extension
    # past the padded extent — windows are truncated there (onnx
    # op_pool_common limits taps to in + pads).
    divisor: Value | None = None
    if any(begins) or any(ends):
        per_dim = []
        for i in range(s):
            outs = _pool_out_sizes(
                [in_sizes[i] + begins[i] + ends[i]],
                [kernel[i]],
                [strides[i]],
                [dilations[i]],
            )[0]
            o = np.arange(outs)[:, None]
            taps = o * strides[i] + np.arange(kernel[i])[None, :] * dilations[i]
            if count_include_pad:
                valid = taps < in_sizes[i] + begins[i] + orig_ends[i]
            else:
                valid = (taps >= begins[i]) & (taps < begins[i] + in_sizes[i])
            per_dim.append(valid.sum(axis=1).astype(np.float32))
        counts = per_dim[0]
        for d in per_dim[1:]:
            counts = np.multiply.outer(counts, d)
        if (counts != math.prod(kernel)).any():
            divisor = coreai.cast(
                coreai.constant(counts.reshape((1, 1, *counts.shape))),
                tensor_type(x).element_type,
            )

    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    is_1d = s == 1
    if is_1d:
        x = coreai.expand_dims(x, [2])
        kernel, strides, dilations = [1, *kernel], [1, *strides], [1, *dilations]
        begins, ends = [0, *begins], [0, *ends]
    x = _pad_spatial(x, begins, ends, zero)
    sumpool = coreai.sumpool2d if len(kernel) == 2 else coreai.sumpool3d
    result = sumpool(
        x,
        kernel_size=np.array(kernel, np.uint32),
        strides=np.array(strides, np.uint32),
        dilation=coreai.constant(dilations, dtype=np.uint32),
    )
    if is_1d:
        result = coreai.shrink_dims(result, [2])
    if divisor is None:
        divisor = coreai.cast(float(math.prod(kernel)), tensor_type(x).element_type)
    return coreai.broadcasting_divide(result, divisor)


def _lp_pool_powered_input(x: Value, p: int) -> Value:
    elem = tensor_type(x).element_type
    if p == 1:
        return coreai.abs_(x)
    if p == 2:
        return coreai.broadcasting_mul(x, x)
    return coreai.broadcasting_pow(
        coreai.abs_(x), coreai.constant(float(p), dtype=elem)
    )


def _lp_pool_root(x: Value, p: int) -> Value:
    if p == 1:
        return x
    if p == 2:
        return coreai.sqrt(x)
    return coreai.broadcasting_pow(
        x, coreai.constant(1.0 / float(p), dtype=tensor_type(x).element_type)
    )


def replace_lp_pool(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, s, node_attrs, kernel, strides, dilations = _pool_setup(
        "LpPool", values_map, node
    )
    if not isinstance(tensor_type(x).element_type, FloatType):
        raise ValueError(
            "LpPool: non-float input dtypes are not supported, got element type "
            f"{tensor_type(x).element_type}"
        )
    p = int(node_attrs.get("p", 2))
    if p <= 0:
        raise ValueError(f"LpPool: p must be positive, got {p}")

    in_sizes = _static_spatial(x, "LpPool")
    begins, ends = _resolve_pads(
        "LpPool", node_attrs, in_sizes, kernel, strides, dilations
    )
    if node_attrs.get("ceil_mode", 0):
        ends = _ceil_mode_ends(in_sizes, kernel, strides, dilations, begins, ends)

    x = _lp_pool_powered_input(x, p)
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    is_1d = s == 1
    if is_1d:
        x = coreai.expand_dims(x, [2])
        kernel, strides, dilations = [1, *kernel], [1, *strides], [1, *dilations]
        begins, ends = [0, *begins], [0, *ends]
    x = _pad_spatial(x, begins, ends, zero)
    sumpool = coreai.sumpool2d if len(kernel) == 2 else coreai.sumpool3d
    result = sumpool(
        x,
        kernel_size=np.array(kernel, np.uint32),
        strides=np.array(strides, np.uint32),
        dilation=coreai.constant(dilations, dtype=np.uint32),
    )
    if is_1d:
        result = coreai.shrink_dims(result, [2])
    return _lp_pool_root(result, p)


def replace_global_lp_pool(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    rank = tensor_type(x).rank
    if rank < 3:
        raise ValueError(f"GlobalLpPool: input rank must be >= 3, got {rank}")
    if not isinstance(tensor_type(x).element_type, FloatType):
        raise ValueError(
            "GlobalLpPool: non-float input dtypes are not supported, got element type "
            f"{tensor_type(x).element_type}"
        )
    p = int(attrs(node).get("p", 2))
    if p <= 0:
        raise ValueError(f"GlobalLpPool: p must be positive, got {p}")
    pooled = coreai.reduce_sum(_lp_pool_powered_input(x, p), list(range(2, rank)))
    return _lp_pool_root(pooled, p)


# ---------------------------------------------------------------------------
# GlobalAveragePool / GlobalMaxPool / GlobalLpPool
# (reduce over spatial dims, dims kept)
# ---------------------------------------------------------------------------


def _global_pool(op_name: str, reduce_fn: Callable[..., Any]) -> Callable[..., Value]:
    def _lower(
        values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
    ) -> Value:
        (x,) = operands(values_map, node, [0])
        if tensor_type(x).rank < 3:
            raise ValueError(
                f"{op_name}: input rank must be >= 3, got {tensor_type(x).rank}"
            )
        # coreai reduce_* keep the reduced dims as size 1, matching ONNX here.
        return reduce_fn(x, list(range(2, tensor_type(x).rank)))

    _lower.__name__ = f"replace_{op_name}"
    return _lower


# ---------------------------------------------------------------------------
# LRN:  y = x / (bias + alpha/size * sum(x^2 over channel window))^beta
# ---------------------------------------------------------------------------


def replace_lrn(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    node_attrs = attrs(node)
    size = node_attrs.get("size")
    if size is None:
        raise ValueError("LRN: the required 'size' attribute is missing")
    size = int(size)
    alpha = float(node_attrs.get("alpha", 1e-4))
    beta = float(node_attrs.get("beta", 0.75))
    bias = float(node_attrs.get("bias", 1.0))
    rank = tensor_type(x).rank
    channels = tensor_type(x).shape[1]
    if rank < 3 or channels < 0:
        raise ValueError("LRN: input must be NCHW-like with a static channel dim")

    elem = tensor_type(x).element_type
    # Zero-pad the channel axis, then take the sliding-window sum as a sum of
    # `size` shifted slices (size is small in practice).
    before = (size - 1) // 2
    padding = [0, 0, before, size - 1 - before] + [0, 0] * (rank - 2)
    squares = coreai.pad(
        coreai.broadcasting_mul(x, x),
        np.array(padding, dtype=np.uint32),
        coreai.constant(0, dtype=elem),
    )
    window_sum = None
    for i in range(size):
        sl = coreai.slice_(
            squares,
            [0, i] + [0] * (rank - 2),
            [_INT32_MAX, i + channels] + [_INT32_MAX] * (rank - 2),
            [1] * rank,
        )
        window_sum = (
            sl if window_sum is None else coreai.broadcasting_add(window_sum, sl)
        )
    denom = coreai.broadcasting_pow(
        coreai.broadcasting_add(
            coreai.constant(bias, dtype=elem),
            coreai.broadcasting_mul(
                coreai.constant(alpha / size, dtype=elem), window_sum
            ),
        ),
        coreai.constant(beta, dtype=elem),
    )
    return coreai.broadcasting_divide(x, denom)


REGISTRY: dict[str, Callable[..., Any]] = {
    "Conv": replace_conv,
    "ConvTranspose": replace_conv_transpose,
    "MaxPool": replace_max_pool,
    "AveragePool": replace_average_pool,
    "LpPool": replace_lp_pool,
    "GlobalAveragePool": _global_pool("GlobalAveragePool", coreai.reduce_mean),
    "GlobalMaxPool": _global_pool("GlobalMaxPool", coreai.reduce_max),
    "GlobalLpPool": replace_global_lp_pool,
    "LRN": replace_lrn,
}
