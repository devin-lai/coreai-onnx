# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for convolution, pooling, and LRN op lowerings."""

import zlib

import numpy as np
import pytest

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import (
    COREAI_RUNTIME_MARKS,
    assert_parity,
    requires_coreai_runtime,
    run_aimodel,
    run_onnxruntime,
    single_op_model,
)

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]

TOL = {"rtol": 1e-3, "atol": 1e-3}


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _rand(key: str, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(_seed(key))
    return (rng.standard_normal(shape) * 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------

CONV2D_CASES = {
    # id: (weight_shape, attrs, with_bias)
    "k1": ((8, 4, 1, 1), {}, False),
    "k3": ((8, 4, 3, 3), {}, False),
    "k3-bias": ((8, 4, 3, 3), {}, True),
    "stride2": ((8, 4, 3, 3), {"strides": [2, 2]}, False),
    "pads-sym": ((8, 4, 3, 3), {"pads": [1, 1, 1, 1]}, True),
    "pads-asym": ((8, 4, 3, 3), {"pads": [0, 1, 0, 1]}, False),
    "same-upper": ((8, 4, 3, 3), {"auto_pad": "SAME_UPPER", "strides": [2, 2]}, False),
    "same-lower": ((8, 4, 3, 3), {"auto_pad": "SAME_LOWER", "strides": [2, 2]}, False),
    "valid": ((8, 4, 3, 3), {"auto_pad": "VALID"}, False),
    "groups2": ((8, 2, 3, 3), {"group": 2}, True),
    "depthwise": ((4, 1, 3, 3), {"group": 4}, False),
    "dilation2": ((8, 4, 3, 3), {"dilations": [2, 2]}, False),
}


@pytest.mark.parametrize("case", CONV2D_CASES.values(), ids=CONV2D_CASES.keys())
async def test_conv2d(case):
    w_shape, attrs, with_bias = case
    key = f"conv2d-{w_shape}-{attrs}-{with_bias}"
    x = _rand(f"{key}-x", (1, 4, 8, 8))
    initializers = {"W": _rand(f"{key}-w", w_shape)}
    if with_bias:
        initializers["B"] = _rand(f"{key}-b", (w_shape[0],))
    model = single_op_model("Conv", {"x": x}, attrs=attrs, initializers=initializers)
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv1d():
    x = _rand("conv1d-x", (1, 3, 16))
    initializers = {
        "W": _rand("conv1d-w", (6, 3, 3)),
        "B": _rand("conv1d-b", (6,)),
    }
    model = single_op_model(
        "Conv",
        {"x": x},
        attrs={"pads": [1, 1], "strides": [2]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv3d():
    x = _rand("conv3d-x", (1, 2, 4, 6, 6))
    initializers = {"W": _rand("conv3d-w", (4, 2, 2, 2, 2))}
    model = single_op_model(
        "Conv",
        {"x": x},
        attrs={"pads": [1, 1, 1, 1, 1, 1], "strides": [2, 2, 2]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


@pytest.mark.skip(
    reason=(
        "Core AI runtime fails to load float16 conv programs (Program load "
        "failure 0x10004) — runtime limitation, not a lowering bug; same "
        "tracked Core AI f16 issue as tests/test_ops_unary.py"
    )
)
async def test_conv2d_f16():
    rng = np.random.default_rng(_seed("conv2d-f16"))
    x = (rng.standard_normal((1, 3, 8, 8)) * 0.5).astype(np.float16)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.5).astype(np.float16)
    model = single_op_model("Conv", {"x": x}, initializers={"W": w})
    await assert_parity(model, {"x": x}, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# ConvTranspose
# ---------------------------------------------------------------------------


async def test_conv_transpose2d_stride2_output_padding_bias():
    x = _rand("convt2d-x", (1, 3, 5, 5))
    initializers = {
        "W": _rand("convt2d-w", (3, 4, 3, 3)),
        "B": _rand("convt2d-b", (4,)),
    }
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2, 2], "output_padding": [1, 1]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv_transpose2d_pads():
    x = _rand("convt2d-pads-x", (1, 3, 5, 5))
    initializers = {"W": _rand("convt2d-pads-w", (3, 4, 3, 3))}
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2, 2], "pads": [1, 1, 1, 1]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


def _conv_transpose_model_no_inference(attrs, out_shape):
    """A one-node ConvTranspose model built without strict shape inference
    (which rejects the auto_pad/pads attribute conflict before the lowering
    ever sees it)."""
    from onnx import TensorProto, helper, numpy_helper

    w = _rand("convt2d-manual-w", (3, 4, 3, 3))
    graph = helper.make_graph(
        [helper.make_node("ConvTranspose", ["x", "W"], ["out0"], **attrs)],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, list(out_shape))],
        initializer=[numpy_helper.from_array(w, "W")],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )


async def test_conv_transpose2d_auto_pad_valid():
    x = _rand("convt2d-valid-x", (1, 3, 5, 5))
    initializers = {"W": _rand("convt2d-valid-w", (3, 4, 3, 3))}
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"auto_pad": "VALID", "strides": [2, 2]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv_transpose2d_auto_pad_valid_ignores_conflicting_pads():
    # auto_pad != NOTSET takes precedence over the pads attribute. ONNX forbids
    # setting both (onnxruntime refuses to load such a model), so the converter
    # must not silently apply the pads as crops; the output must match the
    # equivalent VALID-only model.
    x = _rand("convt2d-validpads-x", (1, 3, 5, 5))
    model = _conv_transpose_model_no_inference(
        {"auto_pad": "VALID", "pads": [1, 1, 1, 1]}, (1, 4, 7, 7)
    )
    clean = _conv_transpose_model_no_inference({"auto_pad": "VALID"}, (1, 4, 7, 7))
    (expected,) = run_onnxruntime(clean, {"x": x})
    (got,) = await run_aimodel(model, {"x": x})
    np.testing.assert_allclose(got, expected, **TOL)


def test_conv_transpose_unknown_auto_pad_rejected():
    # Conv rejects unrecognized auto_pad values via _resolve_pads; ConvTranspose
    # must do the same instead of silently treating them as NOTSET.
    model = _conv_transpose_model_no_inference({"auto_pad": "SAME"}, (1, 4, 7, 7))
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="auto_pad"):
        converter.to_coreai()


async def test_conv_transpose2d_output_shape():
    x = _rand("convt2d-oshape-x", (1, 3, 5, 5))
    initializers = {"W": _rand("convt2d-oshape-w", (3, 4, 3, 3))}
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2, 2], "output_shape": [10, 10]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv_transpose3d():
    x = _rand("convt3d-x", (1, 2, 3, 4, 4))
    initializers = {"W": _rand("convt3d-w", (2, 3, 2, 2, 2))}
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2, 2, 2]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


CONVT2D_GEOMETRY_CASES = {
    # id: (x_shape, w_shape, attrs) — odd 5x5 input with stride 2 makes the
    # SAME_UPPER/SAME_LOWER padding split unevenly
    "group2": ((1, 4, 5, 5), (4, 2, 3, 3), {"group": 2, "strides": [2, 2]}),
    "depthwise-g4": ((1, 4, 5, 5), (4, 1, 3, 3), {"group": 4, "strides": [2, 2]}),
    "same-upper": (
        (1, 3, 5, 5),
        (3, 4, 3, 3),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
    "same-lower": (
        (1, 3, 5, 5),
        (3, 4, 3, 3),
        {"auto_pad": "SAME_LOWER", "strides": [2, 2]},
    ),
    "same-upper-outpad": (
        (1, 3, 5, 5),
        (3, 4, 3, 3),
        {"auto_pad": "SAME_UPPER", "strides": [2, 2], "output_padding": [1, 1]},
    ),
    "dilation2": ((1, 3, 5, 5), (3, 4, 3, 3), {"dilations": [2, 2]}),
    "dilation2-s2-pads": (
        (1, 3, 5, 5),
        (3, 4, 3, 3),
        {"dilations": [2, 2], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    ),
    "group2-same-upper": (
        (1, 4, 5, 5),
        (4, 2, 3, 3),
        {"group": 2, "auto_pad": "SAME_UPPER", "strides": [2, 2]},
    ),
}


@pytest.mark.parametrize(
    "case", CONVT2D_GEOMETRY_CASES.values(), ids=CONVT2D_GEOMETRY_CASES.keys()
)
async def test_conv_transpose2d_geometry(case):
    x_shape, w_shape, attrs = case
    key = f"convt2d-geo-{w_shape}-{attrs}"
    x = _rand(f"{key}-x", x_shape)
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs=attrs,
        initializers={"W": _rand(f"{key}-w", w_shape)},
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv_transpose1d():
    x = _rand("convt1d-x", (1, 3, 16))
    initializers = {
        "W": _rand("convt1d-w", (3, 4, 3)),
        "B": _rand("convt1d-b", (4,)),
    }
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2], "pads": [1, 1], "output_padding": [1]},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_conv_transpose1d_output_shape():
    x = _rand("convt1d-oshape-x", (1, 3, 8))
    model = single_op_model(
        "ConvTranspose",
        {"x": x},
        attrs={"strides": [2], "output_shape": [16]},
        initializers={"W": _rand("convt1d-oshape-w", (3, 4, 3))},
    )
    await assert_parity(model, {"x": x}, **TOL)


# ---------------------------------------------------------------------------
# MaxPool
# ---------------------------------------------------------------------------

MAXPOOL2D_CASES = {
    "k2-s2": {"kernel_shape": [2, 2], "strides": [2, 2]},
    "k3-s1-pads1": {"kernel_shape": [3, 3], "strides": [1, 1], "pads": [1, 1, 1, 1]},
    "same-upper": {"kernel_shape": [2, 2], "strides": [2, 2], "auto_pad": "SAME_UPPER"},
    "dilation2": {"kernel_shape": [2, 2], "dilations": [2, 2]},
}


@pytest.mark.parametrize("attrs", MAXPOOL2D_CASES.values(), ids=MAXPOOL2D_CASES.keys())
async def test_maxpool2d(attrs):
    # 7x7 input so SAME_UPPER needs uneven padding
    x = _rand(f"maxpool2d-{attrs}", (1, 4, 7, 7))
    model = single_op_model("MaxPool", {"x": x}, attrs=attrs)
    await assert_parity(model, {"x": x}, **TOL)


async def test_maxpool1d():
    x = _rand("maxpool1d-x", (1, 3, 16))
    model = single_op_model(
        "MaxPool", {"x": x}, attrs={"kernel_shape": [3], "strides": [2]}
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_maxpool3d():
    x = _rand("maxpool3d-x", (1, 2, 6, 6, 6))
    model = single_op_model(
        "MaxPool", {"x": x}, attrs={"kernel_shape": [2, 2, 2], "strides": [2, 2, 2]}
    )
    await assert_parity(model, {"x": x}, **TOL)


MAXPOOL2D_CEIL_CASES = {
    # id: (input HxW, attrs) — sizes chosen so ceil and floor outputs differ
    "k3-s2": ((6, 6), {"kernel_shape": [3, 3], "strides": [2, 2]}),
    "k3-s2-pads1": (
        (6, 6),
        {"kernel_shape": [3, 3], "strides": [2, 2], "pads": [1, 1, 1, 1]},
    ),
    "k2-s2": ((5, 5), {"kernel_shape": [2, 2], "strides": [2, 2]}),
    # 7x7 k3 s2: ceil == floor (torchvision GoogLeNet's exact maxpool config)
    "k3-s2-noop": ((7, 7), {"kernel_shape": [3, 3], "strides": [2, 2]}),
    "k2-s2-dil2": (
        (6, 6),
        {"kernel_shape": [2, 2], "strides": [2, 2], "dilations": [2, 2]},
    ),
    # stride > window extent: the last ceil-mode window would start in the
    # end padding and must be dropped (ONNX window-start rule)
    "k2-s3-clamp": (
        (3, 3),
        {"kernel_shape": [2, 2], "strides": [3, 3], "pads": [0, 0, 1, 1]},
    ),
}


@pytest.mark.parametrize(
    "case", MAXPOOL2D_CEIL_CASES.values(), ids=MAXPOOL2D_CEIL_CASES.keys()
)
async def test_maxpool2d_ceil_mode(case):
    (h, w), attrs = case
    attrs = dict(attrs, ceil_mode=1)
    x = _rand(f"maxpool2d-ceil-{attrs}", (1, 3, h, w))
    model = single_op_model("MaxPool", {"x": x}, attrs=attrs)
    await assert_parity(model, {"x": x}, **TOL)


async def test_maxpool1d_ceil_mode():
    x = _rand("maxpool1d-ceil-x", (1, 3, 7))
    model = single_op_model(
        "MaxPool", {"x": x}, attrs={"kernel_shape": [2], "strides": [2], "ceil_mode": 1}
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_maxpool3d_ceil_mode():
    x = _rand("maxpool3d-ceil-x", (1, 2, 5, 5, 5))
    model = single_op_model(
        "MaxPool",
        {"x": x},
        attrs={"kernel_shape": [2, 2, 2], "strides": [2, 2, 2], "ceil_mode": 1},
    )
    await assert_parity(model, {"x": x}, **TOL)


def test_maxpool_int8_rejected():
    # ONNX MaxPool-12+ allows int8/uint8 inputs, but the Core AI runtime
    # cannot load integer pooling programs (Program load failure 0x10004) —
    # reject at conversion time instead of shipping an unloadable .aimodel.
    x = np.zeros((1, 2, 6, 6), dtype=np.int8)
    model = single_op_model("MaxPool", {"x": x}, attrs={"kernel_shape": [2, 2]})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="non-float"):
        converter.to_coreai()


def test_maxpool_indices_output_rejected():
    x = np.zeros((1, 4, 8, 8), dtype=np.float32)
    model = single_op_model(
        "MaxPool", {"x": x}, n_outputs=2, attrs={"kernel_shape": [2, 2]}
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match=r"[Ii]ndices"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# AveragePool
# ---------------------------------------------------------------------------

AVGPOOL2D_CASES = {
    "k2-s2": {"kernel_shape": [2, 2], "strides": [2, 2]},
    "k3-pads1-exclude": {
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "count_include_pad": 0,
    },
    "k3-pads1-include": {
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "count_include_pad": 1,
    },
    "same-upper": {"kernel_shape": [2, 2], "strides": [2, 2], "auto_pad": "SAME_UPPER"},
}


@pytest.mark.parametrize("attrs", AVGPOOL2D_CASES.values(), ids=AVGPOOL2D_CASES.keys())
async def test_averagepool2d(attrs):
    x = _rand(f"avgpool2d-{attrs}", (1, 4, 7, 7))
    model = single_op_model("AveragePool", {"x": x}, attrs=attrs)
    await assert_parity(model, {"x": x}, **TOL)


async def test_averagepool1d():
    x = _rand("avgpool1d-x", (1, 3, 16))
    model = single_op_model(
        "AveragePool",
        {"x": x},
        attrs={"kernel_shape": [3], "strides": [2], "pads": [1, 1]},
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_averagepool3d():
    x = _rand("avgpool3d-x", (1, 2, 6, 6, 6))
    model = single_op_model(
        "AveragePool", {"x": x}, attrs={"kernel_shape": [2, 2, 2], "strides": [2, 2, 2]}
    )
    await assert_parity(model, {"x": x}, **TOL)


AVGPOOL2D_CEIL_CASES = {
    # id: (input HxW, attrs) — sizes chosen so ceil and floor outputs differ
    "k3-s2": ((6, 6), {"kernel_shape": [3, 3], "strides": [2, 2]}),
    "k3-s2-pads1-exclude": (
        (6, 6),
        {
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [1, 1, 1, 1],
            "count_include_pad": 0,
        },
    ),
    # include: ceil-mode windows are still truncated at the explicitly padded
    # extent, so the divisor varies near the end borders
    "k3-s2-pads1-include": (
        (6, 6),
        {
            "kernel_shape": [3, 3],
            "strides": [2, 2],
            "pads": [1, 1, 1, 1],
            "count_include_pad": 1,
        },
    ),
    "k4-s3-pads2-include": (
        (5, 5),
        {
            "kernel_shape": [4, 4],
            "strides": [3, 3],
            "pads": [2, 2, 2, 2],
            "count_include_pad": 1,
        },
    ),
}


@pytest.mark.parametrize(
    "case", AVGPOOL2D_CEIL_CASES.values(), ids=AVGPOOL2D_CEIL_CASES.keys()
)
async def test_averagepool2d_ceil_mode(case):
    (h, w), attrs = case
    attrs = dict(attrs, ceil_mode=1)
    x = _rand(f"avgpool2d-ceil-{attrs}", (1, 3, h, w))
    model = single_op_model("AveragePool", {"x": x}, attrs=attrs)
    await assert_parity(model, {"x": x}, **TOL)


async def test_averagepool1d_ceil_mode():
    x = _rand("avgpool1d-ceil-x", (1, 3, 16))
    model = single_op_model(
        "AveragePool",
        {"x": x},
        attrs={
            "kernel_shape": [3],
            "strides": [2],
            "pads": [1, 1],
            "count_include_pad": 1,
            "ceil_mode": 1,
        },
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_averagepool3d_ceil_mode():
    x = _rand("avgpool3d-ceil-x", (1, 2, 5, 5, 5))
    model = single_op_model(
        "AveragePool",
        {"x": x},
        attrs={"kernel_shape": [2, 2, 2], "strides": [2, 2, 2], "ceil_mode": 1},
    )
    await assert_parity(model, {"x": x}, **TOL)


# ---------------------------------------------------------------------------
# LpPool
# ---------------------------------------------------------------------------

LPPOOL2D_CASES = {
    "p1-pads": {
        "kernel_shape": [2, 3],
        "strides": [2, 2],
        "pads": [1, 0, 1, 0],
        "p": 1,
    },
    "p2-same-upper": {
        "kernel_shape": [2, 2],
        "strides": [2, 2],
        "auto_pad": "SAME_UPPER",
        "p": 2,
    },
    "p3-dilation": {
        "kernel_shape": [2, 2],
        "dilations": [2, 2],
        "p": 3,
    },
}


@pytest.mark.parametrize("attrs", LPPOOL2D_CASES.values(), ids=LPPOOL2D_CASES.keys())
async def test_lppool2d(attrs):
    x = _rand(f"lppool2d-{attrs}", (1, 4, 7, 7))
    model = single_op_model("LpPool", {"x": x}, attrs=attrs)
    await assert_parity(model, {"x": x}, **TOL)


async def test_lppool1d():
    x = _rand("lppool1d-x", (1, 3, 16))
    model = single_op_model(
        "LpPool",
        {"x": x},
        attrs={"kernel_shape": [3], "strides": [2], "pads": [1, 1], "p": 2},
    )
    await assert_parity(model, {"x": x}, **TOL)


async def test_lppool3d_ceil_mode():
    x = _rand("lppool3d-ceil-x", (1, 2, 5, 5, 5))
    model = single_op_model(
        "LpPool",
        {"x": x},
        attrs={"kernel_shape": [2, 2, 2], "strides": [2, 2, 2], "ceil_mode": 1},
    )
    await assert_parity(model, {"x": x}, **TOL)


def test_lppool_rejects_nonpositive_p():
    x = np.zeros((1, 2, 6, 6), dtype=np.float32)
    model = single_op_model("LpPool", {"x": x}, attrs={"kernel_shape": [2, 2], "p": 0})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="p must be positive"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# GlobalAveragePool / GlobalMaxPool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["GlobalAveragePool", "GlobalMaxPool"])
async def test_global_pool(op):
    x = _rand(f"global-{op}", (2, 3, 5, 7))
    model = single_op_model(op, {"x": x})
    await assert_parity(model, {"x": x}, **TOL)


@pytest.mark.parametrize("p", [1, 2, 3])
async def test_global_lppool(p):
    x = _rand(f"global-lppool-{p}", (2, 3, 5, 7))
    model = single_op_model("GlobalLpPool", {"x": x}, attrs={"p": p})

    (got,) = await run_aimodel(model, {"x": x})
    expected = np.sum(np.abs(x.astype(np.float64)) ** p, axis=(2, 3), keepdims=True)
    expected = expected ** (1.0 / p)
    np.testing.assert_allclose(got.astype(np.float64), expected, **TOL)


async def test_global_lppool3d_default_p2():
    x = _rand("global-lppool3d", (1, 2, 3, 4, 5))
    model = single_op_model("GlobalLpPool", {"x": x})

    (got,) = await run_aimodel(model, {"x": x})
    expected = np.sqrt(np.sum(x.astype(np.float64) ** 2, axis=(2, 3, 4), keepdims=True))
    np.testing.assert_allclose(got.astype(np.float64), expected, **TOL)


def test_global_lppool_rejects_nonpositive_p():
    x = np.zeros((1, 2, 6, 6), dtype=np.float32)
    model = single_op_model("GlobalLpPool", {"x": x}, attrs={"p": 0})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="p must be positive"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# LRN
# ---------------------------------------------------------------------------


async def test_lrn_size3_defaults():
    x = _rand("lrn3-x", (1, 5, 4, 4))
    model = single_op_model("LRN", {"x": x}, attrs={"size": 3})
    await assert_parity(model, {"x": x}, **TOL)


async def test_lrn_size5_custom_attrs():
    # window wider than half the channel count exercises edge clipping
    # (onnxruntime only accepts odd sizes, so even windows are untestable here)
    x = _rand("lrn5-x", (1, 5, 4, 4))
    model = single_op_model(
        "LRN",
        {"x": x},
        attrs={"size": 5, "alpha": 2e-4, "beta": 0.6, "bias": 1.5},
    )
    await assert_parity(model, {"x": x}, **TOL)
