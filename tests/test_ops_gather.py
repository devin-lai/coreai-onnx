# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for gather/scatter-family and indexing op lowerings (Task 9)."""

import zlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import (
    assert_parity,
    requires_coreai_runtime,
    single_op_model,
    skip_on_compute_unit,
)

pytestmark = [pytest.mark.ops, requires_coreai_runtime]

_skip_gpu_const_output = skip_on_compute_unit(
    "gpu",
    reason="Core AI GPU specializer fails to load any model with a constant-valued output (upstream issue); default/CPU/ANE are fine",
)


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _manual_model(nodes, input_vis, output_names, initializers=None):
    """Dynamic-dim model that single_op_model can't express."""
    initializers = initializers or {}
    graph = helper.make_graph(
        list(nodes),
        "test_manual",
        list(input_vis),
        [
            helper.make_tensor_value_info(n, TensorProto.UNDEFINED, None)
            for n in output_names
        ],
        initializer=[
            onnx.numpy_helper.from_array(a, name=n) for n, a in initializers.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def _assert_conversion_error(model, match):
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match=match):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


async def test_gather_axis0_1d_indices():
    rng = np.random.default_rng(_seed("gather-axis0-1d"))
    x = rng.random((5, 3)).astype(np.float32)
    idx = np.array([0, 3, 1, 4], dtype=np.int64)
    model = single_op_model("Gather", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_axis1_1d_indices():
    rng = np.random.default_rng(_seed("gather-axis1-1d"))
    x = rng.random((3, 5)).astype(np.float32)
    idx = np.array([4, 0, 2], dtype=np.int64)
    model = single_op_model(
        "Gather", {"x": x}, attrs={"axis": 1}, initializers={"idx": idx}
    )
    await assert_parity(model, {"x": x})


async def test_gather_scalar_index():
    rng = np.random.default_rng(_seed("gather-scalar"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = np.array(2, dtype=np.int64)  # rank 0
    model = single_op_model("Gather", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_axis1_scalar_index():
    rng = np.random.default_rng(_seed("gather-axis1-scalar"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = np.array(3, dtype=np.int64)
    model = single_op_model(
        "Gather", {"x": x}, attrs={"axis": 1}, initializers={"idx": idx}
    )
    await assert_parity(model, {"x": x})


async def test_gather_axis1_2d_indices():
    rng = np.random.default_rng(_seed("gather-axis1-2d"))
    x = rng.random((3, 5)).astype(np.float32)
    idx = np.array([[0, 4], [2, 2]], dtype=np.int64)
    model = single_op_model(
        "Gather", {"x": x}, attrs={"axis": 1}, initializers={"idx": idx}
    )
    await assert_parity(model, {"x": x})


async def test_gather_negative_indices():
    rng = np.random.default_rng(_seed("gather-negative"))
    x = rng.random((5, 3)).astype(np.float32)
    idx = np.array([-1, -5, 2], dtype=np.int64)
    model = single_op_model("Gather", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_runtime_indices():
    # Embedding-style lookup: indices arrive as a graph input, not initializer.
    rng = np.random.default_rng(_seed("gather-runtime"))
    x = rng.random((10, 4)).astype(np.float32)
    idx = np.array([[0, 7, -1], [3, -10, 9]], dtype=np.int64)
    model = single_op_model("Gather", {"x": x, "idx": idx})
    await assert_parity(model, {"x": x, "idx": idx})


async def test_gather_runtime_indices_axis1():
    rng = np.random.default_rng(_seed("gather-runtime-axis1"))
    x = rng.random((2, 6)).astype(np.float32)
    idx = np.array([5, -6, 2], dtype=np.int64)
    model = single_op_model("Gather", {"x": x, "idx": idx}, attrs={"axis": 1})
    await assert_parity(model, {"x": x, "idx": idx})


async def test_gather_axis2_rank3_2d_runtime_indices():
    rng = np.random.default_rng(_seed("gather-axis2-rank3-2d"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    idx = np.array([[0, 3, -1], [2, -4, 1]], dtype=np.int64)
    model = single_op_model("Gather", {"x": x, "idx": idx}, attrs={"axis": 2})
    await assert_parity(model, {"x": x, "idx": idx})


async def test_gather_runtime_indices_dynamic_axis_dim():
    # Indexed dim is dynamic AND indices arrive at runtime: the negative-index
    # fix-up must read the dim via get_shape instead of raising.
    rng = np.random.default_rng(_seed("gather-runtime-dynamic"))
    x = rng.random((5, 4)).astype(np.float32)
    idx = np.array([0, -1, 3, -5], dtype=np.int64)
    model = _manual_model(
        [helper.make_node("Gather", ["x", "idx"], ["out0"])],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4]),
            helper.make_tensor_value_info("idx", TensorProto.INT64, [4]),
        ],
        ["out0"],
    )
    await assert_parity(model, {"x": x, "idx": idx})


async def test_gather_const_negative_indices_dynamic_axis_dim():
    rng = np.random.default_rng(_seed("gather-const-neg-dynamic"))
    x = rng.random((5, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Gather", ["x", "idx"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
        initializers={"idx": np.array([-1, 0, -5], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# GatherElements
# ---------------------------------------------------------------------------


async def test_gather_elements_axis0():
    rng = np.random.default_rng(_seed("gather-elements-axis0"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = rng.integers(-3, 3, size=(2, 4)).astype(np.int64)
    model = single_op_model("GatherElements", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_elements_axis1():
    rng = np.random.default_rng(_seed("gather-elements-axis1"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = rng.integers(-4, 4, size=(3, 2)).astype(np.int64)
    model = single_op_model(
        "GatherElements", {"x": x}, attrs={"axis": 1}, initializers={"idx": idx}
    )
    await assert_parity(model, {"x": x})


async def test_gather_elements_runtime_indices():
    rng = np.random.default_rng(_seed("gather-elements-runtime"))
    x = rng.random((4, 5)).astype(np.float32)
    idx = rng.integers(-4, 4, size=(2, 5)).astype(np.int64)
    model = single_op_model("GatherElements", {"x": x, "idx": idx})
    await assert_parity(model, {"x": x, "idx": idx})


# ---------------------------------------------------------------------------
# GatherND
# ---------------------------------------------------------------------------


async def test_gather_nd_full_depth():
    rng = np.random.default_rng(_seed("gather-nd-full"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    idx = np.array(
        [[0, 1, 2], [1, 0, 0], [1, 2, 3], [0, 0, 0], [1, 1, 1]], dtype=np.int64
    )
    model = single_op_model("GatherND", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_nd_partial_depth_negative():
    rng = np.random.default_rng(_seed("gather-nd-partial"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    idx = np.array([[0, -1], [-2, 1]], dtype=np.int64)  # output (2, 4)
    model = single_op_model("GatherND", {"x": x}, initializers={"idx": idx})
    await assert_parity(model, {"x": x})


async def test_gather_nd_batch_dims1():
    rng = np.random.default_rng(_seed("gather-nd-batch1"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    idx = np.array([[[2], [0]], [[1], [2]]], dtype=np.int64)  # (2, 2, 1) -> (2, 2, 4)
    model = single_op_model(
        "GatherND", {"x": x}, attrs={"batch_dims": 1}, initializers={"idx": idx}
    )
    await assert_parity(model, {"x": x})


async def test_gather_nd_runtime_indices():
    rng = np.random.default_rng(_seed("gather-nd-runtime"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = np.array([[0, -1], [2, 2], [-3, 0]], dtype=np.int64)
    model = single_op_model("GatherND", {"x": x, "idx": idx})
    await assert_parity(model, {"x": x, "idx": idx})


async def test_gather_nd_runtime_indices_dynamic_dims():
    # Mixed static/dynamic indexed dims with runtime indices: the dims vector
    # for the negative-index fix-up is read from get_shape at runtime.
    rng = np.random.default_rng(_seed("gather-nd-runtime-dynamic"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = np.array([[0, -1], [-3, 2], [2, 0]], dtype=np.int64)
    model = _manual_model(
        [helper.make_node("GatherND", ["x", "idx"], ["out0"])],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4]),
            helper.make_tensor_value_info("idx", TensorProto.INT64, [3, 2]),
        ],
        ["out0"],
    )
    await assert_parity(model, {"x": x, "idx": idx})


# ---------------------------------------------------------------------------
# ScatterElements
# ---------------------------------------------------------------------------


async def test_scatter_elements_axis0():
    rng = np.random.default_rng(_seed("scatter-elements-axis0"))
    x = rng.random((3, 4)).astype(np.float32)
    # Unique row positions per column (duplicate handling is undefined).
    idx = np.argsort(rng.random((3, 4)), axis=0)[:2].astype(np.int64)
    upd = rng.random((2, 4)).astype(np.float32)
    model = single_op_model(
        "ScatterElements", {"x": x}, initializers={"idx": idx, "upd": upd}
    )
    await assert_parity(model, {"x": x})


async def test_scatter_elements_axis1_negative():
    rng = np.random.default_rng(_seed("scatter-elements-axis1"))
    x = rng.random((3, 4)).astype(np.float32)
    idx = np.argsort(rng.random((3, 4)), axis=1)[:, :2].astype(np.int64)
    idx[0] -= 4  # negative indices address the same (unique) positions
    upd = rng.random((3, 2)).astype(np.float32)
    model = single_op_model(
        "ScatterElements",
        {"x": x},
        attrs={"axis": 1},
        initializers={"idx": idx, "upd": upd},
    )
    await assert_parity(model, {"x": x})


def test_scatter_elements_reduction_unsupported():
    x = np.zeros((3, 4), dtype=np.float32)
    idx = np.zeros((1, 4), dtype=np.int64)
    upd = np.ones((1, 4), dtype=np.float32)
    model = single_op_model(
        "ScatterElements",
        {"x": x},
        attrs={"reduction": "add"},
        initializers={"idx": idx, "upd": upd},
    )
    _assert_conversion_error(model, "reduction")


# ---------------------------------------------------------------------------
# ScatterND
# ---------------------------------------------------------------------------


async def test_scatter_nd_row_updates():
    rng = np.random.default_rng(_seed("scatter-nd-rows"))
    x = rng.random((4, 4)).astype(np.float32)
    idx = np.array([[0], [2]], dtype=np.int64)
    upd = rng.random((2, 4)).astype(np.float32)
    model = single_op_model(
        "ScatterND", {"x": x}, initializers={"idx": idx, "upd": upd}
    )
    await assert_parity(model, {"x": x})


async def test_scatter_nd_point_updates_negative():
    rng = np.random.default_rng(_seed("scatter-nd-points"))
    x = rng.random((4, 4)).astype(np.float32)
    idx = np.array([[0, 1], [-1, -2], [2, 0]], dtype=np.int64)  # unique cells
    upd = rng.random((3,)).astype(np.float32)
    model = single_op_model(
        "ScatterND", {"x": x}, initializers={"idx": idx, "upd": upd}
    )
    await assert_parity(model, {"x": x})


def test_scatter_nd_reduction_unsupported():
    x = np.zeros((4, 4), dtype=np.float32)
    idx = np.array([[0]], dtype=np.int64)
    upd = np.ones((1, 4), dtype=np.float32)
    model = single_op_model(
        "ScatterND",
        {"x": x},
        attrs={"reduction": "add"},
        initializers={"idx": idx, "upd": upd},
    )
    _assert_conversion_error(model, "reduction")


# ---------------------------------------------------------------------------
# OneHot
# ---------------------------------------------------------------------------


async def test_one_hot_default_axis():
    idx = np.array([[0, -1, 3], [4, 2, -5]], dtype=np.int64)
    model = single_op_model(
        "OneHot",
        {"idx": idx},
        initializers={
            "depth": np.array(5, dtype=np.int64),
            "values": np.array([0.5, 1.5], dtype=np.float32),
        },
    )
    await assert_parity(model, {"idx": idx})


async def test_one_hot_axis0():
    idx = np.array([[0, -1, 3], [4, 2, 1]], dtype=np.int64)
    model = single_op_model(
        "OneHot",
        {"idx": idx},
        attrs={"axis": 0},
        initializers={
            "depth": np.array(5, dtype=np.int64),
            "values": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"idx": idx})


async def test_one_hot_axis1_2d_indices():
    # Middle axis: indices (2, 3) with depth 4 and axis=1 -> output (2, 4, 3).
    idx = np.array([[0, 3, -1], [2, -4, 1]], dtype=np.int32)
    model = single_op_model(
        "OneHot",
        {"idx": idx},
        attrs={"axis": 1},
        initializers={
            "depth": np.array(4, dtype=np.int32),
            "values": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"idx": idx})


async def test_one_hot_int32_indices():
    idx = np.array([0, 2, 4, 1], dtype=np.int32)
    model = single_op_model(
        "OneHot",
        {"idx": idx},
        initializers={
            "depth": np.array(5, dtype=np.int32),
            "values": np.array([0.0, 1.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"idx": idx})


async def test_one_hot_out_of_range_gives_off():
    idx = np.array([0, 7, -9, 4], dtype=np.int64)  # 7 and -9 are out of range
    model = single_op_model(
        "OneHot",
        {"idx": idx},
        initializers={
            "depth": np.array(5, dtype=np.int64),
            "values": np.array([2.0, 3.0], dtype=np.float32),
        },
    )
    await assert_parity(model, {"idx": idx})


# ---------------------------------------------------------------------------
# NonZero  (dynamic output shape)
# ---------------------------------------------------------------------------


async def test_non_zero_float():
    x = np.array([[0.0, 1.5, 0.0, 2.0], [3.0, 0.0, 0.0, 4.0]], dtype=np.float32)
    model = single_op_model("NonZero", {"x": x})
    await assert_parity(model, {"x": x})


async def test_non_zero_bool():
    x = np.array([[True, False, True, False, True], [False, True, False, True, False]])
    model = single_op_model("NonZero", {"x": x})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# EyeLike
# ---------------------------------------------------------------------------


@_skip_gpu_const_output
async def test_eye_like_square():
    x = np.zeros((4, 4), dtype=np.float32)
    model = single_op_model("EyeLike", {"x": x})
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_eye_like_rect_k1():
    x = np.zeros((3, 5), dtype=np.float32)
    model = single_op_model("EyeLike", {"x": x}, attrs={"k": 1})
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_eye_like_rect_k_neg1():
    x = np.zeros((5, 3), dtype=np.float32)
    model = single_op_model("EyeLike", {"x": x}, attrs={"k": -1})
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_eye_like_dtype_override():
    x = np.zeros((3, 5), dtype=np.float32)
    model = single_op_model(
        "EyeLike", {"x": x}, attrs={"dtype": TensorProto.INT32, "k": 1}
    )
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_eye_like_default_dtype_follows_input():
    x = np.zeros((4, 4), dtype=np.int32)
    model = single_op_model("EyeLike", {"x": x})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------------


async def test_compress_axis0():
    rng = np.random.default_rng(_seed("compress-axis0"))
    x = rng.random((4, 3)).astype(np.float32)
    cond = np.array([True, False, True, False])
    model = single_op_model(
        "Compress", {"x": x}, attrs={"axis": 0}, initializers={"cond": cond}
    )
    await assert_parity(model, {"x": x})


async def test_compress_axis1_short_condition():
    rng = np.random.default_rng(_seed("compress-axis1"))
    x = rng.random((3, 4)).astype(np.float32)
    cond = np.array([False, True, True])  # shorter than the axis is allowed
    model = single_op_model(
        "Compress", {"x": x}, attrs={"axis": 1}, initializers={"cond": cond}
    )
    await assert_parity(model, {"x": x})


async def test_compress_no_axis_flattens():
    rng = np.random.default_rng(_seed("compress-flat"))
    x = rng.random((3, 4)).astype(np.float32)
    cond = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.bool_)
    model = single_op_model("Compress", {"x": x}, initializers={"cond": cond})
    await assert_parity(model, {"x": x})


def test_compress_runtime_condition_unsupported():
    x = np.zeros((4, 3), dtype=np.float32)
    cond = np.array([True, False, True, False])
    model = single_op_model("Compress", {"x": x, "cond": cond}, attrs={"axis": 0})
    _assert_conversion_error(model, "condition")


# ---------------------------------------------------------------------------
# GridSample  (2D bilinear/nearest sampling; zeros/border padding)
# ---------------------------------------------------------------------------


def _grid_sample_inputs(C=3, H=5, W=7, OH=4, OW=6, seed="grid"):
    rng = np.random.default_rng(_seed(seed))
    x = rng.standard_normal((1, C, H, W)).astype(np.float32)
    # grid spans [-1.4, 1.4] so out-of-bounds samples exercise the padding rule.
    grid = rng.uniform(-1.4, 1.4, (1, OH, OW, 2)).astype(np.float32)
    return x, grid


@pytest.mark.parametrize("align_corners", [0, 1])
@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
@pytest.mark.parametrize("mode", ["linear", "nearest"])
async def test_grid_sample(mode, padding_mode, align_corners):
    x, grid = _grid_sample_inputs(seed=f"grid-{mode}-{padding_mode}-{align_corners}")
    model = single_op_model(
        "GridSample",
        {"x": x, "grid": grid},
        attrs={
            "mode": mode,
            "padding_mode": padding_mode,
            "align_corners": align_corners,
        },
    )
    await assert_parity(model, {"x": x, "grid": grid})


async def test_grid_sample_batched():
    # N > 1 exercises the per-batch sampling loop.
    rng = np.random.default_rng(_seed("grid-batch"))
    x = rng.standard_normal((2, 4, 6, 8)).astype(np.float32)
    grid = rng.uniform(-1.2, 1.2, (2, 5, 3, 2)).astype(np.float32)
    model = single_op_model(
        "GridSample", {"x": x, "grid": grid}, attrs={"align_corners": 1}
    )
    await assert_parity(model, {"x": x, "grid": grid})


def test_grid_sample_cubic_mode_unsupported():
    x, grid = _grid_sample_inputs()
    model = single_op_model(
        "GridSample", {"x": x, "grid": grid}, attrs={"mode": "cubic"}
    )
    _assert_conversion_error(model, "cubic")


def test_grid_sample_reflection_padding_unsupported():
    x, grid = _grid_sample_inputs()
    model = single_op_model(
        "GridSample", {"x": x, "grid": grid}, attrs={"padding_mode": "reflection"}
    )
    _assert_conversion_error(model, "reflection")
