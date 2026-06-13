# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for ONNX linear quantization ops."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import COREAI_RUNTIME_MARKS, assert_parity, requires_coreai_runtime

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]


def _make_model(nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(
        nodes, "quant_test", inputs, outputs, initializer=list(initializers)
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _quantize_model(*, scale, zero_point=None, attrs=None, shape=(2, 3)):
    attrs = attrs or {}
    node_inputs = ["x", "scale"]
    initializers = [
        numpy_helper.from_array(np.asarray(scale, dtype=np.float32), "scale")
    ]
    if zero_point is not None:
        node_inputs.append("zero_point")
        initializers.append(
            numpy_helper.from_array(np.asarray(zero_point), "zero_point")
        )
    return _make_model(
        [helper.make_node("QuantizeLinear", node_inputs, ["y"], **attrs)],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)],
        initializers,
    )


def _dequantize_model(*, x_dtype, scale, zero_point=None, attrs=None, shape=(2, 3)):
    attrs = attrs or {}
    node_inputs = ["x", "scale"]
    initializers = [
        numpy_helper.from_array(np.asarray(scale, dtype=np.float32), "scale")
    ]
    if zero_point is not None:
        node_inputs.append("zero_point")
        initializers.append(
            numpy_helper.from_array(np.asarray(zero_point), "zero_point")
        )
    return _make_model(
        [helper.make_node("DequantizeLinear", node_inputs, ["y"], **attrs)],
        [helper.make_tensor_value_info("x", x_dtype, shape)],
        [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)],
        initializers,
    )


def _qlinear_matmul_model(
    *,
    a_dtype=TensorProto.UINT8,
    b_dtype=TensorProto.UINT8,
    a_scale=0.5,
    a_zero_point=None,
    b_scale=0.25,
    b_zero_point=None,
    y_scale=0.125,
    y_zero_point=None,
    a_shape=(2, 3),
    b_shape=(3, 2),
):
    if a_zero_point is None:
        a_zero_point = np.array(3, dtype=np.uint8)
    if b_zero_point is None:
        b_zero_point = np.array(2, dtype=np.uint8)
    if y_zero_point is None:
        y_zero_point = np.array(10, dtype=np.uint8)
    initializers = [
        numpy_helper.from_array(np.asarray(a_scale, dtype=np.float32), "a_scale"),
        numpy_helper.from_array(np.asarray(a_zero_point), "a_zero_point"),
        numpy_helper.from_array(np.asarray(b_scale, dtype=np.float32), "b_scale"),
        numpy_helper.from_array(np.asarray(b_zero_point), "b_zero_point"),
        numpy_helper.from_array(np.asarray(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(y_zero_point), "y_zero_point"),
    ]
    return _make_model(
        [
            helper.make_node(
                "QLinearMatMul",
                [
                    "a",
                    "a_scale",
                    "a_zero_point",
                    "b",
                    "b_scale",
                    "b_zero_point",
                    "y_scale",
                    "y_zero_point",
                ],
                ["y"],
            )
        ],
        [
            helper.make_tensor_value_info("a", a_dtype, a_shape),
            helper.make_tensor_value_info("b", b_dtype, b_shape),
        ],
        [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)],
        initializers,
    )


def _qlinear_conv_model(
    *,
    x_dtype=TensorProto.UINT8,
    weight=None,
    x_shape=(1, 1, 3, 3),
    x_scale=0.25,
    x_zero_point=None,
    w_scale=0.5,
    w_zero_point=None,
    y_scale=0.125,
    y_zero_point=None,
    bias=None,
    attrs=None,
):
    attrs = attrs or {}
    if weight is None:
        weight = np.array([[[[2, 3], [4, 5]]], [[[6, 7], [8, 9]]]], dtype=np.uint8)
    if x_zero_point is None:
        x_zero_point = np.array(3, dtype=np.uint8)
    if w_zero_point is None:
        w_zero_point = np.array(2, dtype=np.uint8)
    if y_zero_point is None:
        y_zero_point = np.array(7, dtype=np.uint8)
    node_inputs = [
        "x",
        "x_scale",
        "x_zero_point",
        "w",
        "w_scale",
        "w_zero_point",
        "y_scale",
        "y_zero_point",
    ]
    initializers = [
        numpy_helper.from_array(np.asarray(x_scale, dtype=np.float32), "x_scale"),
        numpy_helper.from_array(np.asarray(x_zero_point), "x_zero_point"),
        numpy_helper.from_array(np.asarray(weight), "w"),
        numpy_helper.from_array(np.asarray(w_scale, dtype=np.float32), "w_scale"),
        numpy_helper.from_array(np.asarray(w_zero_point), "w_zero_point"),
        numpy_helper.from_array(np.asarray(y_scale, dtype=np.float32), "y_scale"),
        numpy_helper.from_array(np.asarray(y_zero_point), "y_zero_point"),
    ]
    if bias is not None:
        node_inputs.append("B")
        initializers.append(
            numpy_helper.from_array(np.asarray(bias, dtype=np.int32), "B")
        )
    return _make_model(
        [helper.make_node("QLinearConv", node_inputs, ["y"], **attrs)],
        [helper.make_tensor_value_info("x", x_dtype, x_shape)],
        [helper.make_tensor_value_info("y", TensorProto.UNDEFINED, None)],
        initializers,
    )


@pytest.mark.ir
def test_quantize_linear_lowers_to_coreai_quantize():
    text = str(
        coreai_onnx.convert(
            _quantize_model(scale=0.5, zero_point=np.array(2, dtype=np.uint8))
        )
    )
    assert "coreai.quantize" in text
    assert "tensor<ui8>" in text


@pytest.mark.ir
def test_dequantize_linear_lowers_to_coreai_dequantize():
    text = str(
        coreai_onnx.convert(
            _dequantize_model(
                x_dtype=TensorProto.UINT8,
                scale=0.5,
                zero_point=np.array(2, dtype=np.uint8),
            )
        )
    )
    assert "coreai.dequantize" in text
    assert "tensor<2x3xf32>" in text


@pytest.mark.ir
def test_qlinear_matmul_lowers_to_dequantize_matmul_quantize():
    text = str(coreai_onnx.convert(_qlinear_matmul_model()))
    assert "coreai.dequantize" in text
    assert "coreai.decomposable.broadcasting_batch_matmul" in text
    assert "coreai.quantize" in text
    assert "tensor<2x2xui8>" in text


@pytest.mark.ir
def test_qlinear_conv_lowers_to_dequantize_conv_quantize():
    text = str(
        coreai_onnx.convert(
            _qlinear_conv_model(
                w_scale=np.array([0.5, 0.25], dtype=np.float32),
                w_zero_point=np.array([2, 2], dtype=np.uint8),
                bias=np.array([1, -2], dtype=np.int32),
                attrs={"pads": [1, 1, 1, 1]},
            )
        )
    )
    assert "coreai.dequantize" in text
    assert "coreai.conv2d" in text
    assert "coreai.quantize" in text
    assert "tensor<1x2x4x4xui8>" in text


@pytest.mark.ir
def test_quantize_linear_rejects_unsupported_output_dtype():
    model = _quantize_model(scale=0.5, attrs={"output_dtype": TensorProto.INT16})
    with pytest.raises(ConversionError, match="output_dtype INT16"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_dequantize_linear_rejects_int32_input_before_save_asset():
    model = _dequantize_model(x_dtype=TensorProto.INT32, scale=0.25)
    with pytest.raises(ConversionError, match="int8/uint8"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_quantize_linear_rejects_blocked_quantization():
    model = _quantize_model(
        scale=np.ones((2, 2), dtype=np.float32),
        zero_point=np.zeros((2, 2), dtype=np.uint8),
        attrs={"axis": 1, "block_size": 2},
        shape=(2, 4),
    )
    with pytest.raises(ConversionError, match="blocked quantization"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_qlinear_matmul_rejects_non_scalar_quantization_parameters():
    model = _qlinear_matmul_model(
        a_scale=np.array([0.5, 0.25], dtype=np.float32),
    )
    with pytest.raises(ConversionError, match="a_scale must be scalar"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_qlinear_conv_rejects_non_scalar_input_quantization_parameters():
    model = _qlinear_conv_model(
        x_scale=np.array([0.25, 0.5], dtype=np.float32),
    )
    with pytest.raises(ConversionError, match="x_scale must be scalar"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_qlinear_conv_rejects_mismatched_weight_quantization_parameters():
    model = _qlinear_conv_model(
        w_scale=np.array([0.5, 0.25], dtype=np.float32),
        w_zero_point=np.array(2, dtype=np.uint8),
    )
    with pytest.raises(ConversionError, match="w_scale and w_zero_point"):
        coreai_onnx.convert(model)


@requires_coreai_runtime
async def test_quantize_linear_per_tensor_uint8_parity():
    x = np.array([[-1.0, -0.25, 0.0], [0.25, 1.0, 10.0]], dtype=np.float32)
    model = _quantize_model(scale=0.5, zero_point=np.array(2, dtype=np.uint8))
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_quantize_linear_default_uint8_zero_point_parity():
    x = np.array([[0.0, 0.25, 1.0], [2.0, 4.0, 8.0]], dtype=np.float32)
    model = _quantize_model(scale=0.25)
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_quantize_linear_output_dtype_int8_parity():
    x = np.array([[-3.0, -1.0, 0.0], [1.0, 3.0, 20.0]], dtype=np.float32)
    model = _quantize_model(scale=0.5, attrs={"output_dtype": TensorProto.INT8})
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_quantize_linear_per_axis_int8_parity():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    model = _quantize_model(
        scale=np.array([0.5, 1.0, 2.0], dtype=np.float32),
        zero_point=np.array([0, 1, 2], dtype=np.int8),
        attrs={"axis": 1},
    )
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_dequantize_linear_per_tensor_uint8_parity():
    x = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)
    model = _dequantize_model(
        x_dtype=TensorProto.UINT8,
        scale=0.5,
        zero_point=np.array(2, dtype=np.uint8),
    )
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_dequantize_linear_per_axis_int8_negative_axis_parity():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
    model = _dequantize_model(
        x_dtype=TensorProto.INT8,
        scale=np.array([0.5, 1.0, 2.0], dtype=np.float32),
        zero_point=np.array([0, 1, 2], dtype=np.int8),
        attrs={"axis": -1},
    )
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_dequantize_linear_default_zero_point_parity():
    x = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    model = _dequantize_model(x_dtype=TensorProto.UINT8, scale=0.25)
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_qlinear_matmul_uint8_parity():
    a = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    b = np.array([[3, 4], [5, 6], [7, 8]], dtype=np.uint8)
    await assert_parity(_qlinear_matmul_model(), {"a": a, "b": b})


@requires_coreai_runtime
async def test_qlinear_matmul_int8_parity():
    a = np.array([[-4, -2, 0], [2, 4, 6]], dtype=np.int8)
    b = np.array([[2, 4], [6, 8], [10, 12]], dtype=np.int8)
    model = _qlinear_matmul_model(
        a_dtype=TensorProto.INT8,
        b_dtype=TensorProto.INT8,
        a_scale=0.25,
        a_zero_point=np.array(0, dtype=np.int8),
        b_scale=0.5,
        b_zero_point=np.array(0, dtype=np.int8),
        y_scale=0.125,
        y_zero_point=np.array(-2, dtype=np.int8),
    )
    await assert_parity(model, {"a": a, "b": b})


@requires_coreai_runtime
async def test_qlinear_conv_uint8_per_channel_weight_parity():
    x = np.array([[[[3, 5, 7], [9, 11, 13], [15, 17, 19]]]], dtype=np.uint8)
    model = _qlinear_conv_model(
        weight=np.array([[[[2, 3], [4, 5]]], [[[6, 8], [10, 12]]]], dtype=np.uint8),
        w_scale=np.array([0.5, 0.25], dtype=np.float32),
        w_zero_point=np.array([2, 2], dtype=np.uint8),
        bias=np.array([1, -2], dtype=np.int32),
    )
    await assert_parity(model, {"x": x})


@requires_coreai_runtime
async def test_qlinear_conv_int8_parity():
    x = np.array([[[[-4, -2, 0], [2, 4, 6], [8, 10, 12]]]], dtype=np.int8)
    model = _qlinear_conv_model(
        x_dtype=TensorProto.INT8,
        weight=np.array([[[[-2, 0], [2, 4]]], [[[6, 8], [10, 12]]]], dtype=np.int8),
        x_scale=0.25,
        x_zero_point=np.array(0, dtype=np.int8),
        w_scale=0.5,
        w_zero_point=np.array(0, dtype=np.int8),
        y_scale=0.125,
        y_zero_point=np.array(-3, dtype=np.int8),
        bias=np.array([0, 2], dtype=np.int32),
    )
    await assert_parity(model, {"x": x})
