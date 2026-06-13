# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for normalization ops, Dropout, and Resize lowerings."""

import zlib

import numpy as np
import onnx
import pytest

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import (
    COREAI_RUNTIME_MARKS,
    assert_parity,
    requires_coreai_runtime,
    run_aimodel,
    single_op_model,
)

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _expect_conversion_error(model, match):
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match=match):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# BatchNormalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [None, 1e-3])
async def test_batch_norm(eps):
    rng = np.random.default_rng(_seed(f"batch-norm-{eps}"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    initializers = {
        "scale": rng.standard_normal(3).astype(np.float32),
        "b": rng.standard_normal(3).astype(np.float32),
        "mean": rng.standard_normal(3).astype(np.float32),
        "var": (0.5 + rng.random(3)).astype(np.float32),
    }
    attrs = {} if eps is None else {"epsilon": eps}
    model = single_op_model(
        "BatchNormalization", {"x": x}, attrs=attrs, initializers=initializers
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


def test_batch_norm_const_stats_folds_to_mul_add():
    # With constant stats BN must lower to x*A + B (one mul + one add): the
    # sub-first emission triples full-tensor work and never matches the
    # compiler's conv-scale fusion pattern (FuseConvAndScalingOp).
    rng = np.random.default_rng(_seed("batch-norm-fold"))
    x = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    initializers = {
        "scale": rng.standard_normal(3).astype(np.float32),
        "b": rng.standard_normal(3).astype(np.float32),
        "mean": rng.standard_normal(3).astype(np.float32),
        "var": (0.5 + rng.random(3)).astype(np.float32),
    }
    model = single_op_model("BatchNormalization", {"x": x}, initializers=initializers)
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    text = str(converter.to_coreai())
    assert "broadcasting_sub" not in text
    assert "rsqrt" not in text
    assert text.count("broadcasting_mul") == 1
    assert text.count("broadcasting_add") == 1


def test_batch_norm_training_mode_rejected():
    rng = np.random.default_rng(_seed("batch-norm-training"))
    x = rng.standard_normal((1, 3, 4, 4)).astype(np.float32)
    initializers = {
        "scale": np.ones(3, dtype=np.float32),
        "b": np.zeros(3, dtype=np.float32),
        "mean": np.zeros(3, dtype=np.float32),
        "var": np.ones(3, dtype=np.float32),
    }
    # training_mode=1 requires the running mean/var outputs to be declared.
    model = single_op_model(
        "BatchNormalization",
        {"x": x},
        n_outputs=3,
        attrs={"training_mode": 1},
        initializers=initializers,
    )
    _expect_conversion_error(model, "training_mode")


# ---------------------------------------------------------------------------
# InstanceNormalization
# ---------------------------------------------------------------------------


async def test_instance_norm():
    rng = np.random.default_rng(_seed("instance-norm"))
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    initializers = {
        "scale": rng.standard_normal(3).astype(np.float32),
        "b": rng.standard_normal(3).astype(np.float32),
    }
    model = single_op_model(
        "InstanceNormalization",
        {"x": x},
        attrs={"epsilon": 1e-5},
        initializers=initializers,
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# LayerNormalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [-1, 1, 2])
@pytest.mark.parametrize("with_bias", [True, False])
async def test_layer_norm(axis, with_bias):
    rng = np.random.default_rng(_seed(f"layer-norm-{axis}-{with_bias}"))
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    normalized_shape = x.shape[axis if axis >= 0 else axis + x.ndim :]
    initializers = {"scale": rng.standard_normal(normalized_shape).astype(np.float32)}
    if with_bias:
        initializers["bias"] = rng.standard_normal(normalized_shape).astype(np.float32)
    model = single_op_model(
        "LayerNormalization", {"x": x}, attrs={"axis": axis}, initializers=initializers
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


def test_layer_norm_extra_outputs_rejected():
    rng = np.random.default_rng(_seed("layer-norm-extra-outputs"))
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "LayerNormalization",
        {"x": x},
        n_outputs=3,
        initializers={"scale": np.ones(4, dtype=np.float32)},
    )
    _expect_conversion_error(model, "output")


# ---------------------------------------------------------------------------
# GroupNormalization  (opset 22: per-channel scale/bias of shape [C])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("groups", [1, 3])
async def test_group_norm(groups):
    rng = np.random.default_rng(_seed(f"group-norm-{groups}"))
    x = rng.standard_normal((2, 6, 4, 4)).astype(np.float32)
    initializers = {
        "scale": rng.standard_normal(6).astype(np.float32),
        "bias": rng.standard_normal(6).astype(np.float32),
    }
    model = single_op_model(
        "GroupNormalization",
        {"x": x},
        attrs={"num_groups": groups, "epsilon": 1e-5},
        initializers=initializers,
    )
    # This onnx build has no shape inference for GroupNormalization; ORT and
    # the checker reject an UNDEFINED output type, so declare it explicitly.
    model.graph.output[0].CopyFrom(
        onnx.helper.make_tensor_value_info("out0", onnx.TensorProto.FLOAT, x.shape)
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


def test_group_norm_target_shape_uses_coreai_values():
    rng = np.random.default_rng(_seed("group-norm-target-shape"))
    x = rng.standard_normal((2, 6, 4, 4)).astype(np.float32)
    model = single_op_model(
        "GroupNormalization",
        {"x": x},
        attrs={"num_groups": 3},
        initializers={
            "scale": np.ones(6, dtype=np.float32),
            "bias": np.zeros(6, dtype=np.float32),
        },
    )
    model.graph.output[0].CopyFrom(
        onnx.helper.make_tensor_value_info("out0", onnx.TensorProto.FLOAT, x.shape)
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)

    text = str(converter.to_coreai())

    assert "coreai.concat" in text
    assert "coreai.reshape" in text


# ---------------------------------------------------------------------------
# LpNormalization / MeanVarianceNormalization
# ---------------------------------------------------------------------------


def _lp_normalize_ref(x, p, axis):
    if p == 1:
        norm = np.sum(np.abs(x), axis=axis, keepdims=True)
    else:
        norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    # Zero vectors normalize to 0 — the lowering's defined behavior (it guards the
    # divide), which we check directly because ONNX Runtime ships no
    # LpNormalization(22) CPU kernel to serve as a parity reference.
    return np.where(norm == 0, 0.0, x / np.where(norm == 0, 1.0, norm))


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("axis", [1, -1])
async def test_lp_normalization(p, axis):
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    model = single_op_model("LpNormalization", {"x": x}, attrs={"p": p, "axis": axis})
    (got,) = await run_aimodel(model, {"x": x})
    np.testing.assert_allclose(got, _lp_normalize_ref(x, p, axis), rtol=1e-3, atol=1e-4)


def test_lp_normalization_rejects_unsupported_p():
    x = np.ones((2, 3), dtype=np.float32)
    model = single_op_model("LpNormalization", {"x": x}, attrs={"p": 3})
    _expect_conversion_error(model, "p=3")


def _mvn_model(shape, attrs=None):
    # onnx 1.21 shape inference expands MeanVarianceNormalization's function
    # body to a malformed Constant node. Declare the output type explicitly.
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "MeanVarianceNormalization", ["x"], ["y"], **(attrs or {})
            )
        ],
        "mvn",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 22)], ir_version=10
    )


@pytest.mark.parametrize("attrs", [None, {"axes": [1, 3]}])
async def test_mean_variance_normalization(attrs):
    x = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    x = (x % 7 - 3) / 3
    model = _mvn_model(x.shape, attrs)
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Dropout  (inference: identity)
# ---------------------------------------------------------------------------


async def test_dropout_plain():
    rng = np.random.default_rng(_seed("dropout-plain"))
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    model = single_op_model("Dropout", {"x": x})
    await assert_parity(model, {"x": x})


async def test_dropout_with_ratio():
    rng = np.random.default_rng(_seed("dropout-ratio"))
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "Dropout", {"x": x}, initializers={"ratio": np.array(0.3, dtype=np.float32)}
    )
    await assert_parity(model, {"x": x})


def test_dropout_training_mode_true_rejected():
    rng = np.random.default_rng(_seed("dropout-training"))
    x = rng.standard_normal((2, 3)).astype(np.float32)
    model = single_op_model(
        "Dropout",
        {"x": x},
        initializers={
            "ratio": np.array(0.5, dtype=np.float32),
            "training_mode": np.array(True),
        },
    )
    _expect_conversion_error(model, "training_mode")


def test_dropout_mask_output_rejected():
    rng = np.random.default_rng(_seed("dropout-mask"))
    x = rng.standard_normal((2, 3)).astype(np.float32)
    model = single_op_model("Dropout", {"x": x}, n_outputs=2)
    _expect_conversion_error(model, "mask")


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

_EMPTY_ROI = np.array([], dtype=np.float32)
_EMPTY_SCALES = np.array([], dtype=np.float32)


async def test_resize_nearest_scales_x2():
    rng = np.random.default_rng(_seed("resize-nearest-x2"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_linear_half_pixel_x2():
    rng = np.random.default_rng(_seed("resize-linear-x2"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_linear_align_corners():
    rng = np.random.default_rng(_seed("resize-linear-align-corners"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "coordinate_transformation_mode": "align_corners"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_linear_sizes():
    rng = np.random.default_rng(_seed("resize-linear-sizes"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": _EMPTY_SCALES,
            "sizes": np.array([1, 3, 15, 13], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_nearest_asymmetric():
    rng = np.random.default_rng(_seed("resize-nearest-asymmetric"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest", "coordinate_transformation_mode": "asymmetric"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_nearest_axes_spatial_only():
    rng = np.random.default_rng(_seed("resize-nearest-axes"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest", "axes": [2, 3]},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_nearest_axes_unsorted_scales():
    # Opset 18+: scales follow the ORDER of `axes`, so axes=[3, 2] with
    # scales=[2, 3] means W*2 and H*3 — (1,1,4,8) -> (1,1,12,16).
    rng = np.random.default_rng(_seed("resize-nearest-axes-unsorted"))
    x = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest", "axes": [3, 2]},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([2.0, 3.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_linear_axes_unsorted_sizes():
    # sizes are also given in `axes` order: axes=[3, 2], sizes=[16, 12].
    rng = np.random.default_rng(_seed("resize-linear-axes-unsorted"))
    x = rng.standard_normal((1, 1, 4, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "axes": [3, 2]},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": _EMPTY_SCALES,
            "sizes": np.array([16, 12], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


_NEAREST_MODES = ["round_prefer_floor", "round_prefer_ceil", "floor", "ceil"]
_CT_MODES = ["half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"]


@pytest.mark.parametrize("coord_mode", _CT_MODES)
@pytest.mark.parametrize("nearest_mode", _NEAREST_MODES)
async def test_resize_nearest_matrix(coord_mode, nearest_mode):
    # x2 hits exact .5 ties, where each nearest_mode differs.
    rng = np.random.default_rng(_seed(f"resize-near-{coord_mode}-{nearest_mode}"))
    x = rng.standard_normal((1, 2, 4, 5)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={
            "mode": "nearest",
            "coordinate_transformation_mode": coord_mode,
            "nearest_mode": nearest_mode,
        },
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("coord_mode", _CT_MODES)
@pytest.mark.parametrize("scale", [0.5, 1.5, 2.0])
async def test_resize_linear_matrix(coord_mode, scale):
    rng = np.random.default_rng(_seed(f"resize-lin-{coord_mode}-{scale}"))
    x = rng.standard_normal((1, 2, 4, 6)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "coordinate_transformation_mode": coord_mode},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, scale, scale], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_nearest_sizes():
    rng = np.random.default_rng(_seed("resize-nearest-sizes"))
    x = rng.standard_normal((1, 2, 4, 6)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": _EMPTY_SCALES,
            "sizes": np.array([1, 2, 9, 5], dtype=np.int64),  # up H, down W
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


async def test_resize_nearest_downscale():
    rng = np.random.default_rng(_seed("resize-nearest-downscale"))
    x = rng.standard_normal((1, 2, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "nearest"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 0.5, 0.75], dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


def test_resize_cubic_rejected():
    rng = np.random.default_rng(_seed("resize-cubic"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "cubic"},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    _expect_conversion_error(model, "cubic")


def test_resize_antialias_rejected():
    rng = np.random.default_rng(_seed("resize-antialias"))
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "antialias": 1},
        initializers={
            "roi": _EMPTY_ROI,
            "scales": np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32),
        },
    )
    _expect_conversion_error(model, "antialias")


# ---------------------------------------------------------------------------
# float16 coverage  (LayerNormalization exercises stash_type=1: stage-one
# statistics must be computed/stored in float32 even for fp16 inputs)
# ---------------------------------------------------------------------------


async def test_f16_layer_norm_stash_type():
    # Mean far from zero: storing the statistics in fp16 deviates from the
    # spec fp32 stage-one by ~5e-2 while ORT stays within ~6e-3, so tolerances
    # of 2e-2 fail unless stash_type=1 (default) is honored.
    rng = np.random.default_rng(_seed("f16-layer-norm"))
    x = (8.0 + 0.05 * rng.standard_normal((2, 3, 32))).astype(np.float16)
    initializers = {
        "scale": rng.standard_normal(32).astype(np.float16),
        "bias": rng.standard_normal(32).astype(np.float16),
    }
    model = single_op_model(
        "LayerNormalization", {"x": x}, attrs={"axis": -1}, initializers=initializers
    )
    await assert_parity(model, {"x": x}, rtol=2e-2, atol=2e-2)


@pytest.mark.skip(
    reason=(
        "Core AI runtime crashes (Program load failure 0x10004) for float16 "
        "inputs to Softmax — not a lowering bug; tracked as a Core AI f16 "
        "issue (same as Relu/Sigmoid/Softplus in test_ops_unary.py)"
    )
)
async def test_f16_softmax():
    rng = np.random.default_rng(_seed("f16-softmax"))
    x = (rng.standard_normal((4, 16)) * 4.0).astype(np.float16)
    model = single_op_model("Softmax", {"x": x})
    await assert_parity(model, {"x": x}, rtol=1e-2, atol=1e-2)
