# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for shape & data-movement op lowerings (Task 7)."""

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

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _manual_model(nodes, input_vis, output_names, initializers=None, opset=22):
    """Multi-node / dynamic-dim model that single_op_model can't express."""
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
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


# ---------------------------------------------------------------------------
# Reshape
# ---------------------------------------------------------------------------


async def test_reshape_explicit():
    rng = np.random.default_rng(_seed("reshape-explicit"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "Reshape", {"x": x}, initializers={"shape": np.array([4, 6], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_reshape_infer_minus_one():
    rng = np.random.default_rng(_seed("reshape-minus-one"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "Reshape", {"x": x}, initializers={"shape": np.array([-1, 6], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_reshape_zero_copies_input_dim():
    rng = np.random.default_rng(_seed("reshape-zero"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "Reshape", {"x": x}, initializers={"shape": np.array([0, 12], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_reshape_zero_and_minus_one():
    rng = np.random.default_rng(_seed("reshape-zero-minus-one"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "Reshape", {"x": x}, initializers={"shape": np.array([0, -1], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


def _runtime_shape_reshape_model(x_dims, shape_len):
    return _manual_model(
        [helper.make_node("Reshape", ["x", "shape"], ["out0"])],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_dims),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [shape_len]),
        ],
        ["out0"],
    )


async def test_reshape_runtime_shape_zero_copies_dim():
    # allowzero=0: a 0 in the *runtime* shape tensor copies the input dim.
    rng = np.random.default_rng(_seed("reshape-runtime-zero"))
    x = rng.random((2, 3)).astype(np.float32)
    model = _runtime_shape_reshape_model([2, 3], 2)
    await assert_parity(model, {"x": x, "shape": np.array([0, 3], dtype=np.int64)})


async def test_reshape_runtime_shape_zero_lower_rank():
    # Runtime shape shorter than the input rank, with a 0 in head position.
    rng = np.random.default_rng(_seed("reshape-runtime-zero-lo"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = _runtime_shape_reshape_model([2, 3, 4], 2)
    await assert_parity(model, {"x": x, "shape": np.array([0, 12], dtype=np.int64)})


async def test_reshape_runtime_shape_zero_higher_rank():
    # Runtime shape longer than the input rank; the 0 copies input dim 0.
    rng = np.random.default_rng(_seed("reshape-runtime-zero-hi"))
    x = rng.random((2, 3)).astype(np.float32)
    model = _runtime_shape_reshape_model([2, 3], 3)
    await assert_parity(model, {"x": x, "shape": np.array([0, 3, 1], dtype=np.int64)})


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


async def test_transpose_with_perm():
    rng = np.random.default_rng(_seed("transpose-perm"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Transpose", {"x": x}, attrs={"perm": [2, 0, 1]})
    await assert_parity(model, {"x": x})


async def test_transpose_default_reverses_axes():
    rng = np.random.default_rng(_seed("transpose-default"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Transpose", {"x": x})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [1, -1])
async def test_concat_three_inputs(axis):
    rng = np.random.default_rng(_seed(f"concat-{axis}"))
    a = rng.random((2, 2)).astype(np.float32)
    b = rng.random((2, 3)).astype(np.float32)
    c = rng.random((2, 1)).astype(np.float32)
    inputs = {"a": a, "b": b, "c": c}
    model = single_op_model("Concat", inputs, attrs={"axis": axis})
    await assert_parity(model, inputs)


async def test_concat_4d_channel_axis():
    rng = np.random.default_rng(_seed("concat-4d-channel"))
    a = rng.random((1, 2, 4, 4)).astype(np.float32)
    b = rng.random((1, 3, 4, 4)).astype(np.float32)
    c = rng.random((1, 1, 4, 4)).astype(np.float32)
    inputs = {"a": a, "b": b, "c": c}
    model = single_op_model("Concat", inputs, attrs={"axis": 1})
    await assert_parity(model, inputs)


async def test_concat_wide_fanin_survives_runtime():
    """A static concat of 12+ inputs lowered as one pad+where chain makes the
    MPSGraph compiler emit a fused kernel that exceeds Metal's buffer-argument
    table ('Unable to get MPS kernel ndArrayIdentity ... invalid location'),
    aborting the process on the GPU/default unit (seen in stereonet's 27-way
    and SSD's 80-way concats). The lowering must chunk wide concats."""
    rng = np.random.default_rng(_seed("concat-wide"))
    inputs = {
        f"x{i:02d}": rng.random((1, 2, 4, 4)).astype(np.float32) for i in range(27)
    }
    model = single_op_model("Concat", inputs, attrs={"axis": 1})
    await assert_parity(model, inputs)


async def test_concat_wide_fanin_integer_survives_runtime():
    """Same as above for the integer pad+add composite path."""
    rng = np.random.default_rng(_seed("concat-wide-int"))
    inputs = {
        f"x{i:02d}": rng.integers(-9, 9, (1, 2, 4, 4)).astype(np.int32)
        for i in range(27)
    }
    model = single_op_model("Concat", inputs, attrs={"axis": 1})
    await assert_parity(model, inputs)


async def test_concat_preserves_float_signed_zero():
    a = np.array([[-0.0]], dtype=np.float32)
    b = np.array([[0.0]], dtype=np.float32)
    model = _manual_model(
        [
            helper.make_node("Concat", ["a", "b"], ["cat"], axis=1),
            helper.make_node("Reciprocal", ["cat"], ["out0"]),
        ],
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a.shape),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, b.shape),
        ],
        ["out0"],
    )
    await assert_parity(model, {"a": a, "b": b})


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


async def test_split_equal_num_outputs():
    rng = np.random.default_rng(_seed("split-num-outputs"))
    x = rng.random((4, 6)).astype(np.float32)
    model = single_op_model(
        "Split", {"x": x}, n_outputs=2, attrs={"axis": 0, "num_outputs": 2}
    )
    await assert_parity(model, {"x": x})


async def test_split_uneven_num_outputs():
    # dim 5 into 2 outputs: ONNX spec gives [3, 2]
    rng = np.random.default_rng(_seed("split-uneven"))
    x = rng.random((5, 2)).astype(np.float32)
    model = single_op_model(
        "Split", {"x": x}, n_outputs=2, attrs={"axis": 0, "num_outputs": 2}
    )
    await assert_parity(model, {"x": x})


async def test_split_explicit_sizes():
    rng = np.random.default_rng(_seed("split-sizes"))
    x = rng.random((4, 3)).astype(np.float32)
    model = single_op_model(
        "Split",
        {"x": x},
        n_outputs=2,
        attrs={"axis": 0},
        initializers={"split": np.array([1, 3], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_split_negative_axis():
    rng = np.random.default_rng(_seed("split-neg-axis"))
    x = rng.random((2, 6)).astype(np.float32)
    model = single_op_model(
        "Split", {"x": x}, n_outputs=2, attrs={"axis": -1, "num_outputs": 2}
    )
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("dim", [1, 2])
def test_split_num_outputs_exceeding_dim_rejected(dim):
    # ONNX Runtime rejects num_outputs > dim ("Invalid num_outputs value of 3.
    # Size of dimension being split is 1"); without the guard the equal-chunk
    # formula underflows the uint32 section sizes (dim=1 -> [1, 1, -1]).
    model = single_op_model(
        "Split",
        {"x": np.zeros((dim, 2), dtype=np.float32)},
        n_outputs=3,
        attrs={"axis": 0, "num_outputs": 3},
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="num_outputs"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


async def test_slice_basic():
    rng = np.random.default_rng(_seed("slice-basic"))
    x = rng.random((5, 6)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([1, 2], dtype=np.int64),
            "ends": np.array([4, 5], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_negative_bounds():
    rng = np.random.default_rng(_seed("slice-negative"))
    x = rng.random((5, 6)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([-4, -5], dtype=np.int64),
            "ends": np.array([-1, -2], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_axes_subset():
    rng = np.random.default_rng(_seed("slice-axes"))
    x = rng.random((5, 6)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([2], dtype=np.int64),
            "ends": np.array([5], dtype=np.int64),
            "axes": np.array([1], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_steps():
    rng = np.random.default_rng(_seed("slice-steps"))
    x = rng.random((8, 3)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([0], dtype=np.int64),
            "ends": np.array([INT64_MAX], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
            "steps": np.array([2], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_negative_step_full_reverse():
    rng = np.random.default_rng(_seed("slice-reverse"))
    x = rng.random((5, 4)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([-1], dtype=np.int64),
            "ends": np.array([INT64_MIN], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
            "steps": np.array([-1], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_negative_step_partial():
    rng = np.random.default_rng(_seed("slice-reverse-partial"))
    x = rng.random((6, 2)).astype(np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([4], dtype=np.int64),
            "ends": np.array([0], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
            "steps": np.array([-2], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_last_token_dynamic_axis():
    # starts=[-1], ends=[INT64_MAX] on a dynamic axis: the canonical
    # last-token slice emitted by transformer exporters.
    rng = np.random.default_rng(_seed("slice-last-token"))
    x = rng.random((4, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Slice", ["x", "starts", "ends", "axes"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
        initializers={
            "starts": np.array([-1], dtype=np.int64),
            "ends": np.array([INT64_MAX], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_slice_runtime_starts_ends():
    # starts/ends arrive as graph inputs (runtime tensors), step 1.
    rng = np.random.default_rng(_seed("slice-runtime-bounds"))
    x = rng.random((5, 4)).astype(np.float32)
    starts = np.array([1, -3], dtype=np.int64)
    ends = np.array([10, 3], dtype=np.int64)
    model = _manual_model(
        [helper.make_node("Slice", ["x", "starts", "ends"], ["out0"])],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [5, 4]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
        ],
        ["out0"],
    )
    await assert_parity(model, {"x": x, "starts": starts, "ends": ends})


def test_slice_negative_step_dynamic_axis_unsupported():
    model = _manual_model(
        [helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
        initializers={
            "starts": np.array([-1], dtype=np.int64),
            "ends": np.array([INT64_MIN], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
            "steps": np.array([-1], dtype=np.int64),
        },
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="negative steps"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Squeeze / Unsqueeze
# ---------------------------------------------------------------------------


async def test_squeeze_with_axes():
    rng = np.random.default_rng(_seed("squeeze-axes"))
    x = rng.random((1, 3, 1, 4)).astype(np.float32)
    model = single_op_model(
        "Squeeze", {"x": x}, initializers={"axes": np.array([0, 2], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_squeeze_negative_axes():
    rng = np.random.default_rng(_seed("squeeze-neg"))
    x = rng.random((3, 1, 4)).astype(np.float32)
    model = single_op_model(
        "Squeeze", {"x": x}, initializers={"axes": np.array([-2], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_squeeze_all_unit_dims():
    rng = np.random.default_rng(_seed("squeeze-all"))
    x = rng.random((1, 3, 1, 4)).astype(np.float32)
    model = single_op_model("Squeeze", {"x": x})
    await assert_parity(model, {"x": x})


async def test_unsqueeze():
    rng = np.random.default_rng(_seed("unsqueeze"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "Unsqueeze", {"x": x}, initializers={"axes": np.array([0, 3], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_unsqueeze_negative_axis():
    rng = np.random.default_rng(_seed("unsqueeze-neg"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "Unsqueeze", {"x": x}, initializers={"axes": np.array([-1], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
async def test_flatten(axis):
    rng = np.random.default_rng(_seed(f"flatten-{axis}"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Flatten", {"x": x}, attrs={"axis": axis})
    await assert_parity(model, {"x": x})


async def test_flatten_default_axis():
    rng = np.random.default_rng(_seed("flatten-default"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Flatten", {"x": x})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Expand / Tile
# ---------------------------------------------------------------------------


async def test_expand_rank_extension():
    rng = np.random.default_rng(_seed("expand-rank"))
    x = rng.random((3, 1)).astype(np.float32)
    model = single_op_model(
        "Expand", {"x": x}, initializers={"shape": np.array([2, 3, 4], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_expand_bidirectional():
    # ONNX Expand is numpy-style: a 1 in `shape` keeps the input dim
    rng = np.random.default_rng(_seed("expand-bidir"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "Expand", {"x": x}, initializers={"shape": np.array([2, 1, 4], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


async def test_expand_dynamic_input_const_keepdim_shape():
    # x: ['n', 3] with shape=[1, 3]: bidirectional broadcast keeps the dynamic
    # batch dim. Regression: this used to SIGABRT the process at .aimodel load
    # ('mps.broadcast_to' inferred (2,3) vs declared (1,3) type conflict).
    rng = np.random.default_rng(_seed("expand-dyn-keepdim"))
    x = rng.random((2, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Expand", ["x", "shape"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
        initializers={"shape": np.array([1, 3], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_expand_dynamic_input_lower_rank_shape():
    # `shape` may have lower rank than the input (left-padded with 1s).
    rng = np.random.default_rng(_seed("expand-dyn-lower-rank"))
    x = rng.random((2, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Expand", ["x", "shape"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
        initializers={"shape": np.array([3], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_expand_shape_from_graph_input():
    # 'shape' arrives as a runtime tensor (a BlockArgument): regression for
    # "'Block' object has no attribute 'name'" in the broadcast_to builder.
    rng = np.random.default_rng(_seed("expand-shape-input"))
    x = rng.random((2, 3)).astype(np.float32)
    shape = np.array([2, 2, 3], dtype=np.int64)
    model = _manual_model(
        [helper.make_node("Expand", ["x", "shape"], ["out0"])],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [3]),
        ],
        ["out0"],
    )
    await assert_parity(model, {"x": x, "shape": shape})


async def test_tile():
    rng = np.random.default_rng(_seed("tile"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model(
        "Tile", {"x": x}, initializers={"repeats": np.array([2, 3], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Pad
# ---------------------------------------------------------------------------


async def test_pad_constant():
    rng = np.random.default_rng(_seed("pad-constant"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        initializers={"pads": np.array([1, 0, 2, 1], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_pad_constant_value():
    rng = np.random.default_rng(_seed("pad-value"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        initializers={
            "pads": np.array([0, 1, 1, 0], dtype=np.int64),
            "value": np.array(1.5, dtype=np.float32),
        },
    )
    await assert_parity(model, {"x": x})


async def test_pad_axes_subset():
    rng = np.random.default_rng(_seed("pad-axes"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        initializers={
            "pads": np.array([1, 2], dtype=np.int64),
            "value": np.array(0.0, dtype=np.float32),
            "axes": np.array([1], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


async def test_pad_reflect():
    rng = np.random.default_rng(_seed("pad-reflect"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        attrs={"mode": "reflect"},
        initializers={"pads": np.array([1, 2, 1, 2], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_pad_edge():
    # ONNX 'edge' replicates the border row/column (maps to Core AI 'replicate').
    rng = np.random.default_rng(_seed("pad-edge"))
    x = rng.random((1, 4, 5, 6)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        attrs={"mode": "edge"},
        initializers={"pads": np.array([0, 0, 1, 2, 0, 0, 2, 1], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_pad_wrap():
    # ONNX 'wrap' wraps around the borders (maps to Core AI 'circular').
    rng = np.random.default_rng(_seed("pad-wrap"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        attrs={"mode": "wrap"},
        initializers={"pads": np.array([1, 2, 1, 2], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


def test_pad_unknown_mode_unsupported():
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model(
        "Pad",
        {"x": x},
        attrs={"mode": "symmetric"},
        initializers={"pads": np.array([1, 1, 1, 1], dtype=np.int64)},
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="symmetric"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


async def test_identity_feeding_graph_output():
    rng = np.random.default_rng(_seed("identity"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model("Identity", {"x": x})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Cast / CastLike
# ---------------------------------------------------------------------------


async def test_cast_f32_to_f16():
    rng = np.random.default_rng(_seed("cast-f16"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model("Cast", {"x": x}, attrs={"to": TensorProto.FLOAT16})
    await assert_parity(model, {"x": x})


async def test_cast_f32_to_i32_truncates():
    x = np.array([1.7, -1.7, 0.5, -0.5, 2.0], dtype=np.float32)
    model = single_op_model("Cast", {"x": x}, attrs={"to": TensorProto.INT32})
    await assert_parity(model, {"x": x})


async def test_cast_i32_to_bool():
    x = np.array([0, 1, -2, 0, 7], dtype=np.int32)
    model = single_op_model("Cast", {"x": x}, attrs={"to": TensorProto.BOOL})
    await assert_parity(model, {"x": x})


async def test_cast_bool_to_f32():
    x = np.array([True, False, True], dtype=np.bool_)
    model = single_op_model("Cast", {"x": x}, attrs={"to": TensorProto.FLOAT})
    await assert_parity(model, {"x": x})


async def test_cast_like_f32_to_f16():
    rng = np.random.default_rng(_seed("castlike"))
    x = rng.random((2, 3)).astype(np.float32)
    model = single_op_model(
        "CastLike",
        {"x": x},
        initializers={"target": np.zeros((1,), dtype=np.float16)},
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Shape / Size
# ---------------------------------------------------------------------------


@_skip_gpu_const_output
async def test_shape_full_with_start_attr():
    # the start attr keeps preprocess from folding the node away
    rng = np.random.default_rng(_seed("shape-full"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Shape", {"x": x}, attrs={"start": 0})
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_shape_start_end():
    rng = np.random.default_rng(_seed("shape-start-end"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Shape", {"x": x}, attrs={"start": 1, "end": 3})
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_shape_negative_end():
    rng = np.random.default_rng(_seed("shape-neg-end"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Shape", {"x": x}, attrs={"end": -1})
    await assert_parity(model, {"x": x})


async def test_shape_dynamic_dim():
    rng = np.random.default_rng(_seed("shape-dyn"))
    x = rng.random((5, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Shape", ["x"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_size_static():
    rng = np.random.default_rng(_seed("size"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("Size", {"x": x})
    await assert_parity(model, {"x": x})


async def test_size_dynamic_dim():
    rng = np.random.default_rng(_seed("size-dyn"))
    x = rng.random((5, 3)).astype(np.float32)
    model = _manual_model(
        [helper.make_node("Size", ["x"], ["out0"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 3])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# DepthToSpace / SpaceToDepth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
async def test_depth_to_space(mode):
    rng = np.random.default_rng(_seed(f"d2s-{mode}"))
    x = rng.random((1, 8, 2, 3)).astype(np.float32)
    model = single_op_model(
        "DepthToSpace", {"x": x}, attrs={"blocksize": 2, "mode": mode}
    )
    await assert_parity(model, {"x": x})


async def test_space_to_depth():
    rng = np.random.default_rng(_seed("s2d"))
    x = rng.random((1, 2, 4, 6)).astype(np.float32)
    model = single_op_model("SpaceToDepth", {"x": x}, attrs={"blocksize": 2})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Trilu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("upper", [0, 1])
async def test_trilu_default_k(upper):
    rng = np.random.default_rng(_seed(f"trilu-{upper}"))
    x = rng.random((4, 5)).astype(np.float32)
    model = single_op_model("Trilu", {"x": x}, attrs={"upper": upper})
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("upper", [0, 1])
@pytest.mark.parametrize("k", [1, -1])
async def test_trilu_with_k(upper, k):
    rng = np.random.default_rng(_seed(f"trilu-k-{upper}-{k}"))
    x = rng.random((4, 4)).astype(np.float32)
    model = single_op_model(
        "Trilu",
        {"x": x},
        attrs={"upper": upper},
        initializers={"k": np.array(k, dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


async def test_trilu_batched():
    rng = np.random.default_rng(_seed("trilu-batch"))
    x = rng.random((2, 3, 3)).astype(np.float32)
    model = single_op_model("Trilu", {"x": x}, attrs={"upper": 1})
    await assert_parity(model, {"x": x})


def test_trilu_mask_constant_is_not_dense():
    # The keep-mask must be built from O(rows+cols) iota constants, not a
    # dense rows*cols matrix embedded in the model (4096-context causal masks
    # would otherwise add ~16.8M constant elements per Trilu node).
    x = np.zeros((128, 128), dtype=np.float32)
    model = single_op_model("Trilu", {"x": x}, attrs={"upper": 1})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    program = converter.to_coreai()

    biggest = 0

    def walk(op):
        nonlocal biggest
        if op.name == "coreai.constant":
            dims = list(op.results[0].type.shape)
            biggest = max(biggest, int(np.prod(dims)) if dims else 1)
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner)

    for op in program._mlir_module.body.operations:
        walk(op)
    assert biggest <= 256, f"largest constant has {biggest} elements"


# ---------------------------------------------------------------------------
# Constant / ConstantOfShape
# ---------------------------------------------------------------------------


async def test_constant_value_tensor():
    rng = np.random.default_rng(_seed("constant"))
    x = rng.random((3,)).astype(np.float32)
    const = onnx.numpy_helper.from_array(
        np.array([1.0, 2.0, 3.0], dtype=np.float32), name="cval"
    )
    model = _manual_model(
        [
            helper.make_node("Constant", [], ["c"], value=const),
            helper.make_node("Add", ["x", "c"], ["out0"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_constant_of_shape_static_int():
    # Shape-with-attrs survives preprocess, so the lowering's static path runs.
    rng = np.random.default_rng(_seed("cos-static"))
    x = rng.random((2, 3)).astype(np.float32)
    value = helper.make_tensor("v", TensorProto.INT32, [1], [5])
    model = _manual_model(
        [
            helper.make_node("Shape", ["x"], ["s"], start=0),
            helper.make_node("ConstantOfShape", ["s"], ["out0"], value=value),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


@_skip_gpu_const_output
async def test_constant_of_shape_default_value():
    rng = np.random.default_rng(_seed("cos-default"))
    x = rng.random((2, 4)).astype(np.float32)
    model = _manual_model(
        [
            helper.make_node("Shape", ["x"], ["s"], start=0),
            helper.make_node("ConstantOfShape", ["s"], ["out0"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


async def test_constant_of_shape_dynamic():
    rng = np.random.default_rng(_seed("cos-dynamic"))
    x = rng.random((3, 2)).astype(np.float32)
    value = helper.make_tensor("v", TensorProto.FLOAT, [1], [1.5])
    model = _manual_model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("ConstantOfShape", ["s"], ["out0"], value=value),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 2])],
        ["out0"],
    )
    await assert_parity(model, {"x": x})


async def test_constant_of_shape_shape_from_graph_input():
    # 'input' (the shape) is a runtime tensor (a BlockArgument): regression
    # for "'Block' object has no attribute 'name'" in the broadcast_to builder.
    shape = np.array([2, 3], dtype=np.int64)
    model = _manual_model(
        [helper.make_node("ConstantOfShape", ["s"], ["out0"])],
        [helper.make_tensor_value_info("s", TensorProto.INT64, [2])],
        ["out0"],
    )
    await assert_parity(model, {"s": shape})


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


@_skip_gpu_const_output
async def test_range_static_int():
    model = single_op_model(
        "Range",
        {},
        initializers={
            "start": np.array(0, dtype=np.int32),
            "limit": np.array(10, dtype=np.int32),
            "delta": np.array(2, dtype=np.int32),
        },
    )
    await assert_parity(model, {})


@_skip_gpu_const_output
async def test_range_static_float():
    model = single_op_model(
        "Range",
        {},
        initializers={
            "start": np.array(1.0, dtype=np.float32),
            "limit": np.array(2.5, dtype=np.float32),
            "delta": np.array(0.5, dtype=np.float32),
        },
    )
    await assert_parity(model, {})


async def test_range_dynamic_int():
    inputs = {
        "start": np.array(0, dtype=np.int32),
        "limit": np.array(10, dtype=np.int32),
        "delta": np.array(2, dtype=np.int32),
    }
    model = single_op_model("Range", inputs)
    await assert_parity(model, inputs)


async def test_range_dynamic_float():
    inputs = {
        "start": np.array(0.0, dtype=np.float32),
        "limit": np.array(1.0, dtype=np.float32),
        "delta": np.array(0.25, dtype=np.float32),
    }
    model = single_op_model("Range", inputs)
    await assert_parity(model, inputs)


# ---------------------------------------------------------------------------
# DepthToSpace / SpaceToDepth divisibility validation
# ---------------------------------------------------------------------------


def test_depth_to_space_channel_not_divisible():
    # C=7 is not divisible by blocksize²=4
    x = np.zeros((1, 7, 4, 4), dtype=np.float32)
    model = single_op_model(
        "DepthToSpace", {"x": x}, attrs={"blocksize": 2, "mode": "DCR"}
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="not divisible"):
        converter.to_coreai()


def test_space_to_depth_hw_not_divisible():
    # H=3 is not divisible by blocksize=2
    x = np.zeros((1, 1, 3, 4), dtype=np.float32)
    model = single_op_model("SpaceToDepth", {"x": x}, attrs={"blocksize": 2})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="not divisible"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Slice: empty-result parity (starts == ends on an axis)
# ---------------------------------------------------------------------------


async def test_slice_empty_result():
    x = np.arange(5, dtype=np.float32)
    model = single_op_model(
        "Slice",
        {"x": x},
        initializers={
            "starts": np.array([2], dtype=np.int64),
            "ends": np.array([2], dtype=np.int64),
            "axes": np.array([0], dtype=np.int64),
        },
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Reshape allowzero=1 with a 0-size dimension
# ---------------------------------------------------------------------------


def test_reshape_allowzero_zero_dim():
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model(
        "Reshape",
        {"x": x},
        initializers={"shape": np.array([0, 6], dtype=np.int64)},
        attrs={"allowzero": 1},
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="allowzero"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Pad: duplicate axes
# ---------------------------------------------------------------------------


def test_pad_duplicate_axes():
    # ONNX strict shape inference rejects duplicate axes, so build the model
    # by hand with a pre-populated (plausible) output shape to get past
    # onnx.checker.check_model inside add_onnx_model.
    x_vi = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    pads_init = onnx.numpy_helper.from_array(
        np.array([1, 1, 1, 1], dtype=np.int64), name="pads"
    )
    value_init = onnx.numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="value"
    )
    axes_init = onnx.numpy_helper.from_array(
        np.array([0, 0], dtype=np.int64), name="axes"
    )
    node = helper.make_node(
        "Pad", inputs=["x", "pads", "value", "axes"], outputs=["out0"]
    )
    out_vi = helper.make_tensor_value_info("out0", TensorProto.FLOAT, [4, 3])
    graph = helper.make_graph(
        [node],
        "test_pad_dup_axes",
        [x_vi],
        [out_vi],
        initializer=[pads_init, value_init, axes_init],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="duplicate"):
        converter.to_coreai()
