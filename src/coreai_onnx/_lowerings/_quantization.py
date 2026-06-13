# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for ONNX linear quantization ops."""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto

from .._ir import F16Type, F32Type, IntegerType, Location, Type, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, normalize_axis, operand, operands
from ._conv import _conv_values
from ._matmul import _matmul_values

_OUTPUT_DTYPE_TO_NUMPY: dict[int, type[np.integer]] = {
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
}


def _numpy_dtype(t: Type, op_name: str) -> np.dtype:
    """Numpy dtype for Core AI element types we can materialize as constants."""
    if isinstance(t, F32Type):
        return np.dtype(np.float32)
    if isinstance(t, F16Type):
        return np.dtype(np.float16)
    if isinstance(t, IntegerType):
        if t.width == 8 and t.is_signed:
            return np.dtype(np.int8)
        if t.width == 8 and t.is_unsigned:
            return np.dtype(np.uint8)
        if t.width == 16 and t.is_signed:
            return np.dtype(np.int16)
        if t.width == 16 and t.is_unsigned:
            return np.dtype(np.uint16)
        if t.width == 32 and t.is_signed:
            return np.dtype(np.int32)
    raise ValueError(f"{op_name}: unsupported quantization parameter dtype {t}")


def _quantized_param_dtype(t: Type, op_name: str) -> np.dtype:
    dtype = _numpy_dtype(t, op_name)
    if dtype not in (np.dtype(np.int8), np.dtype(np.uint8)):
        raise ValueError(
            f"{op_name}: Core AI currently supports int8/uint8 linear "
            f"quantization parameters, got {t}"
        )
    return dtype


def _param_shape(param: Value, param_name: str, op_name: str) -> tuple[int, ...]:
    param_type = tensor_type(param)
    if param_type.rank == 0:
        return ()
    if param_type.rank == 1:
        (dim,) = param_type.shape
        if dim < 0:
            raise ValueError(f"{op_name}: {param_name} length must be statically known")
        return (dim,)
    raise ValueError(
        f"{op_name}: blocked quantization is not supported; {param_name} rank is "
        f"{param_type.rank}, expected scalar or 1-D per-axis parameter"
    )


def _zero_param_like(scale: Value, dtype: Type, op_name: str) -> Value:
    shape = _param_shape(scale, "scale", op_name)
    if not shape:
        return coreai.constant(0, dtype=dtype)
    return coreai.constant(np.zeros(shape, dtype=_numpy_dtype(dtype, op_name)))


def _require_matching_param_shapes(
    scale: Value,
    zero_point: Value,
    scale_name: str,
    zero_point_name: str,
    op_name: str,
) -> None:
    scale_shape = _param_shape(scale, scale_name, op_name)
    zero_point_shape = _param_shape(zero_point, zero_point_name, op_name)
    if scale_shape != zero_point_shape:
        raise ValueError(
            f"{op_name}: {scale_name} and {zero_point_name} must have the same shape, "
            f"got {scale_shape or 'scalar'} and {zero_point_shape or 'scalar'}"
        )


def _axis_param(x: Value, scale: Value, axis: int, op_name: str) -> Value:
    scale_rank = tensor_type(scale).rank
    x_rank = tensor_type(x).rank
    if x_rank == 0:
        raise ValueError(f"{op_name}: scalar inputs are not supported")
    if scale_rank == 0:
        return coreai.constant(np.arange(x_rank, dtype=np.int32))
    if scale_rank == 1:
        return coreai.constant(np.array(normalize_axis(axis, x_rank), dtype=np.int32))
    raise ValueError(
        f"{op_name}: blocked quantization is not supported; scale rank is "
        f"{scale_rank}, expected scalar or 1-D per-axis scale"
    )


def _dequantize_linear_value(
    x: Value, scale: Value, zero_point: Value, axis: int, op_name: str
) -> Value:
    _quantized_param_dtype(tensor_type(zero_point).element_type, op_name)
    _require_matching_param_shapes(scale, zero_point, "scale", "zero_point", op_name)
    offset2 = _zero_param_like(scale, tensor_type(scale).element_type, op_name)
    return coreai.dequantize(
        x, scale, zero_point, offset2, _axis_param(x, scale, axis, op_name)
    )


def _quantize_linear_value(
    x: Value, scale: Value, zero_point: Value, axis: int, op_name: str
) -> Value:
    _quantized_param_dtype(tensor_type(zero_point).element_type, op_name)
    _require_matching_param_shapes(scale, zero_point, "scale", "zero_point", op_name)
    offset2 = _zero_param_like(scale, tensor_type(x).element_type, op_name)
    return coreai.quantize(
        x, scale, zero_point, offset2, _axis_param(x, scale, axis, op_name)
    )


def replace_quantize_linear(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, scale = operands(values_map, node, [0, 1])
    zero_point = operand(values_map, node, 2)
    node_attrs = attrs(node)
    if int(node_attrs.get("block_size", 0)) != 0:
        raise ValueError("QuantizeLinear: blocked quantization is not supported")
    output_dtype = int(node_attrs.get("output_dtype", 0))
    if output_dtype not in (0, *tuple(_OUTPUT_DTYPE_TO_NUMPY)):
        dtype_name = TensorProto.DataType.Name(output_dtype)
        raise ValueError(f"QuantizeLinear: output_dtype {dtype_name} is not supported")

    if zero_point is None:
        dtype = _OUTPUT_DTYPE_TO_NUMPY.get(output_dtype, np.uint8)
        zero_point = coreai.constant(
            np.zeros(_param_shape(scale, "scale", "QuantizeLinear"), dtype=dtype)
        )
    else:
        out_type = tensor_type(zero_point).element_type
        out_dtype = _quantized_param_dtype(out_type, "QuantizeLinear")
        if output_dtype and out_dtype != np.dtype(_OUTPUT_DTYPE_TO_NUMPY[output_dtype]):
            raise ValueError(
                "QuantizeLinear: output_dtype must match y_zero_point dtype"
            )

    return _quantize_linear_value(
        x,
        scale,
        zero_point,
        int(node_attrs.get("axis", 1)),
        "QuantizeLinear",
    )


def replace_dequantize_linear(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    x, scale = operands(values_map, node, [0, 1])
    zero_point = operand(values_map, node, 2)
    node_attrs = attrs(node)
    if int(node_attrs.get("block_size", 0)) != 0:
        raise ValueError("DequantizeLinear: blocked quantization is not supported")
    if zero_point is None:
        _quantized_param_dtype(tensor_type(x).element_type, "DequantizeLinear")
        zero_point = _zero_param_like(
            scale, tensor_type(x).element_type, "DequantizeLinear"
        )
    return _dequantize_linear_value(
        x,
        scale,
        zero_point,
        int(node_attrs.get("axis", 1)),
        "DequantizeLinear",
    )


def _require_scalar_qparam(v: Value, name: str, op_name: str) -> None:
    if tensor_type(v).rank != 0:
        raise ValueError(f"{op_name}: {name} must be scalar")


def _require_per_tensor_or_output_channel_qparam(
    v: Value, name: str, out_channels: int, op_name: str
) -> None:
    shape = _param_shape(v, name, op_name)
    if shape and shape != (out_channels,):
        raise ValueError(
            f"{op_name}: {name} must be scalar or length {out_channels}, got {shape}"
        )


def _require_int32_bias(bias: Value, out_channels: int, op_name: str) -> None:
    bias_type = tensor_type(bias)
    if bias_type.rank != 1 or bias_type.shape[0] != out_channels:
        raise ValueError(
            f"{op_name}: bias must be a 1-D tensor with length {out_channels}"
        )
    elem = bias_type.element_type
    if not isinstance(elem, IntegerType) or elem.width != 32 or not elem.is_signed:
        raise ValueError(f"{op_name}: bias must have int32 element type, got {elem}")


def replace_qlinear_matmul(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (
        a,
        a_scale,
        a_zero_point,
        b,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
    ) = operands(values_map, node, list(range(8)))

    for name, value in (
        ("a_scale", a_scale),
        ("a_zero_point", a_zero_point),
        ("b_scale", b_scale),
        ("b_zero_point", b_zero_point),
        ("y_scale", y_scale),
        ("y_zero_point", y_zero_point),
    ):
        _require_scalar_qparam(value, name, "QLinearMatMul")

    a = _dequantize_linear_value(
        a,
        a_scale,
        a_zero_point,
        0,
        "QLinearMatMul",
    )
    b = _dequantize_linear_value(
        b,
        b_scale,
        b_zero_point,
        0,
        "QLinearMatMul",
    )
    out = _matmul_values(a, b, "QLinearMatMul")
    return _quantize_linear_value(out, y_scale, y_zero_point, 0, "QLinearMatMul")


def replace_qlinear_conv(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (
        x,
        x_scale,
        x_zero_point,
        weight,
        weight_scale,
        weight_zero_point,
        y_scale,
        y_zero_point,
    ) = operands(values_map, node, list(range(8)))
    bias = operand(values_map, node, 8)

    out_channels = tensor_type(weight).shape[0]
    if out_channels < 0:
        raise ValueError("QLinearConv: weight output channels must be static")

    for name, value in (
        ("x_scale", x_scale),
        ("x_zero_point", x_zero_point),
        ("y_scale", y_scale),
        ("y_zero_point", y_zero_point),
    ):
        _require_scalar_qparam(value, name, "QLinearConv")
    for name, value in (
        ("w_scale", weight_scale),
        ("w_zero_point", weight_zero_point),
    ):
        _require_per_tensor_or_output_channel_qparam(
            value, name, out_channels, "QLinearConv"
        )
    _require_matching_param_shapes(
        weight_scale,
        weight_zero_point,
        "w_scale",
        "w_zero_point",
        "QLinearConv",
    )

    x = _dequantize_linear_value(x, x_scale, x_zero_point, 0, "QLinearConv")
    weight = _dequantize_linear_value(
        weight, weight_scale, weight_zero_point, 0, "QLinearConv"
    )
    if bias is not None:
        _require_int32_bias(bias, out_channels, "QLinearConv")
        bias = coreai.broadcasting_mul(
            coreai.cast(bias, np.float32),
            coreai.broadcasting_mul(x_scale, weight_scale),
        )

    out = _conv_values(x, weight, bias, attrs(node), "QLinearConv")
    return _quantize_linear_value(out, y_scale, y_zero_point, 0, "QLinearConv")


REGISTRY: dict[str, Callable[..., Any]] = {
    "QuantizeLinear": replace_quantize_linear,
    "DequantizeLinear": replace_dequantize_linear,
    "QLinearMatMul": replace_qlinear_matmul,
    "QLinearConv": replace_qlinear_conv,
}
