# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for reduction, Arg*, CumSum, and TopK op lowerings (Task 8)."""

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
    single_op_model,
)

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Direct reductions: ReduceSum / ReduceMean / ReduceMax / ReduceMin / ReduceProd
# ---------------------------------------------------------------------------

REDUCE_OPS = ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd"]
AXES_CASES = {"all": None, "axis0": [0], "axis-1": [-1], "axes02": [0, 2]}
# BASELINE_OPSET=17: real opset-17 models carry axes as an attribute, opset 18+
# as input 1 — both forms are live production paths and must stay covered.
AXES_FORMS = ["input", "attr"]


def _axes_model(op, x, axes, keepdims, axes_form):
    attrs = {"keepdims": keepdims}
    initializers = {}
    opset = 22
    if axes_form == "attr":
        opset = 17
        if axes is not None:
            attrs["axes"] = axes
    elif axes is not None:
        initializers["axes"] = np.array(axes, dtype=np.int64)
    return single_op_model(
        op, {"x": x}, attrs=attrs, initializers=initializers, opset=opset
    )


@pytest.mark.parametrize("op", REDUCE_OPS)
@pytest.mark.parametrize("axes", AXES_CASES.values(), ids=AXES_CASES.keys())
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_direct(op, axes, keepdims, axes_form):
    if axes_form == "attr" and op == "ReduceSum":
        pytest.skip(
            "ReduceSum moved axes to an input at opset 13; no opset-17 attr form"
        )
    rng = np.random.default_rng(_seed(f"{op}-{axes}-{keepdims}"))
    # values near 1 so ReduceProd over up to 24 elements stays well-conditioned
    x = (0.75 + 0.5 * rng.random((2, 3, 4))).astype(np.float32)
    model = _axes_model(op, x, axes, keepdims, axes_form)
    await assert_parity(model, {"x": x})


def test_reduce_sum_duplicate_axes_rejected():
    # axes=[0, -3] on a rank-3 tensor both normalize to 0 → duplicate.
    x = np.zeros((2, 3, 4), dtype=np.float32)
    model = single_op_model(
        "ReduceSum",
        {"x": x},
        initializers={"axes": np.array([0, -3], dtype=np.int64)},
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="duplicate"):
        converter.to_coreai()


async def test_reduce_sum_empty_axes_tensor_reduces_all():
    # An explicit empty int64 axes initializer with noop_with_empty_axes absent
    # (default 0) should reduce over all axes, not be a no-op.
    rng = np.random.default_rng(_seed("reduce-sum-empty-axes-tensor"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model(
        "ReduceSum",
        {"x": x},
        attrs={"keepdims": 0},
        initializers={"axes": np.array([], dtype=np.int64)},
    )
    # onnxruntime also treats empty axes (noop_with_empty_axes=0) as reduce-all,
    # so assert_parity is valid here.
    await assert_parity(model, {"x": x})


async def test_reduce_sum_noop_with_empty_axes():
    # Spec: absent axes + noop_with_empty_axes=1 is the identity.  onnxruntime
    # has a kernel bug here (it reduces all axes and then crashes), so compare
    # against the input directly instead of using assert_parity.
    rng = np.random.default_rng(_seed("reduce-sum-noop"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    model = single_op_model("ReduceSum", {"x": x}, attrs={"noop_with_empty_axes": 1})
    (got,) = await run_aimodel(model, {"x": x})
    np.testing.assert_array_equal(np.asarray(got), x)


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean", "ReduceLogSumExp"])
async def test_reduce_scalar_input_keepdims0(op):
    # Rank-0 input with default axes reduces over zero axes; keepdims=0 used
    # to emit shrink_dims(result, []) which hard-aborted (SIGABRT) at convert.
    x = np.array(3.5, dtype=np.float32)
    model = single_op_model(op, {"x": x}, attrs={"keepdims": 0})
    await assert_parity(model, {"x": x})


async def test_reduce_sum_int32():
    rng = np.random.default_rng(_seed("reduce-sum-int32"))
    x = rng.integers(-50, 50, size=(2, 3, 4)).astype(np.int32)
    model = single_op_model(
        "ReduceSum", {"x": x}, initializers={"axes": np.array([1], dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Derived reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_l1(keepdims, axes_form):
    rng = np.random.default_rng(_seed(f"reduce-l1-{keepdims}"))
    x = (rng.random((2, 3, 4)) * 2.0 - 1.0).astype(np.float32)  # negative values
    model = _axes_model("ReduceL1", x, [1], keepdims, axes_form)
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_l2(keepdims, axes_form):
    rng = np.random.default_rng(_seed(f"reduce-l2-{keepdims}"))
    x = (rng.random((2, 3, 4)) * 2.0 - 1.0).astype(np.float32)
    model = _axes_model("ReduceL2", x, [1], keepdims, axes_form)
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_log_sum(keepdims, axes_form):
    rng = np.random.default_rng(_seed(f"reduce-logsum-{keepdims}"))
    x = (rng.random((2, 3, 4)) + 0.5).astype(np.float32)  # strictly positive
    model = _axes_model("ReduceLogSum", x, [1], keepdims, axes_form)
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_log_sum_exp_stability(keepdims, axes_form):
    rng = np.random.default_rng(_seed(f"reduce-lse-{keepdims}"))
    x = rng.random((2, 3, 4)).astype(np.float32)
    x[:, 0, :] += 100.0  # naive exp() overflows float32 without max-shift
    model = _axes_model("ReduceLogSumExp", x, [1], keepdims, axes_form)
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes_form", AXES_FORMS)
async def test_reduce_sum_square(keepdims, axes_form):
    rng = np.random.default_rng(_seed(f"reduce-sumsq-{keepdims}"))
    x = (rng.random((2, 3, 4)) * 2.0 - 1.0).astype(np.float32)
    model = _axes_model("ReduceSumSquare", x, [1], keepdims, axes_form)
    await assert_parity(model, {"x": x})


def _noop_elementwise_ref(op, x):
    if op == "ReduceL1":
        return np.abs(x)
    if op == "ReduceL2":
        return np.sqrt(x * x)
    if op == "ReduceLogSum":
        return np.log(x)
    return x * x  # ReduceSumSquare


@pytest.mark.parametrize(
    "op", ["ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceSumSquare"]
)
async def test_reduce_noop_empty_axes_applies_elementwise(op):
    # Absent axes + noop_with_empty_axes=1 reduces over ZERO axes, so the
    # elementwise pre/post still applies: |x|, sqrt(x*x), log(x), x*x — not the
    # identity. onnxruntime 1.23.x rejects this model (ReduceL1 etc. raise
    # "ValidateNoTransposeReduce ... Reduction on all axes, output size should be
    # 1"), so we validate against a numpy reference run through the Core AI
    # runtime instead of assert_parity.
    rng = np.random.default_rng(_seed(f"reduce-noop-elementwise-{op}"))
    x = (rng.random((2, 3)) + 0.5).astype(np.float32)  # positive for log
    if op != "ReduceLogSum":
        x *= rng.choice([-1.0, 1.0], size=x.shape).astype(np.float32)
    model = single_op_model(op, {"x": x}, attrs={"noop_with_empty_axes": 1})
    (got,) = await run_aimodel(model, {"x": x})
    np.testing.assert_allclose(got, _noop_elementwise_ref(op, x), rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# ArgMax / ArgMin  (distinct values: ties are ill-defined for parity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("axis", [0, -1])
@pytest.mark.parametrize("keepdims", [0, 1])
async def test_arg_ops(op, axis, keepdims):
    rng = np.random.default_rng(_seed(f"{op}-{axis}-{keepdims}"))
    x = rng.permutation(24).astype(np.float32).reshape(2, 3, 4) - 12.0
    model = single_op_model(op, {"x": x}, attrs={"axis": axis, "keepdims": keepdims})
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("op", ["ArgMax", "ArgMin"])
def test_arg_select_last_index_unsupported(op):
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model(op, {"x": x}, attrs={"select_last_index": 1})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="select_last_index"):
        converter.to_coreai()


async def test_argmin_unsigned_input():
    # ONNX ArgMin supports unsigned tensors. The lowering must preserve their
    # ordering instead of using a negation-based transformation.
    x = np.array([[3, 1, 2], [0, 5, 4]], dtype=np.uint8)
    model = single_op_model("ArgMin", {"x": x}, attrs={"axis": 1, "keepdims": 1})
    await assert_parity(model, {"x": x})


async def test_argmin_int32_min_value():
    # Negating INT_MIN overflows, so this catches regressions to the old
    # ArgMax(-x) lowering.
    x = np.array(
        [
            [np.iinfo(np.int32).min, -1, 0],
            [5, np.iinfo(np.int32).min, 1],
        ],
        dtype=np.int32,
    )
    model = single_op_model("ArgMin", {"x": x}, attrs={"axis": 1, "keepdims": 1})
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("op", ["ArgMax", "ArgMin"])
async def test_arg_ops_int32(op):
    """ArgMax/ArgMin parity for int32 inputs (not just float32)."""
    rng = np.random.default_rng(_seed(f"{op}-int32"))
    x = rng.permutation(24).astype(np.int32).reshape(2, 3, 4) - 12
    model = single_op_model(op, {"x": x}, attrs={"axis": 1, "keepdims": 1})
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# CumSum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.int32], ids=["f32", "i32"])
async def test_cumsum_basic(axis, dtype):
    rng = np.random.default_rng(_seed(f"cumsum-{axis}-{np.dtype(dtype).name}"))
    if dtype == np.float32:
        x = rng.random((3, 4)).astype(np.float32)
    else:
        x = rng.integers(-20, 20, size=(3, 4)).astype(np.int32)
    model = single_op_model(
        "CumSum", {"x": x}, initializers={"axis": np.array(axis, dtype=np.int64)}
    )
    await assert_parity(model, {"x": x})


@pytest.mark.parametrize("exclusive", [0, 1])
@pytest.mark.parametrize("reverse", [0, 1])
async def test_cumsum_flags(exclusive, reverse):
    rng = np.random.default_rng(_seed(f"cumsum-flags-{exclusive}-{reverse}"))
    x = rng.random((3, 4)).astype(np.float32)
    model = single_op_model(
        "CumSum",
        {"x": x},
        attrs={"exclusive": exclusive, "reverse": reverse},
        initializers={"axis": np.array(1, dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# TopK  (two outputs: values + indices; distinct values to avoid tie ambiguity)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("largest", [1, 0])
async def test_topk(largest):
    rng = np.random.default_rng(_seed(f"topk-{largest}"))
    x = rng.permutation(15).astype(np.float32).reshape(3, 5)
    model = single_op_model(
        "TopK",
        {"x": x},
        n_outputs=2,
        attrs={"axis": -1, "largest": largest},
        initializers={"k": np.array([2], dtype=np.int64)},
    )
    await assert_parity(model, {"x": x})


def test_topk_k_exceeds_dim_rejected():
    # Build the model with explicit output shapes so onnx.checker passes even
    # though k=6 > dim=5 — the guard lives in the converter, not ONNX.
    from onnx import TensorProto, helper, numpy_helper

    k_init = numpy_helper.from_array(np.array([6], dtype=np.int64), name="k")
    node = helper.make_node(
        "TopK",
        inputs=["x", "k"],
        outputs=["values", "indices"],
        axis=-1,
    )
    graph = helper.make_graph(
        [node],
        "test_topk_k_exceeds",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 5])],
        [
            helper.make_tensor_value_info("values", TensorProto.FLOAT, [3, 6]),
            helper.make_tensor_value_info("indices", TensorProto.INT64, [3, 6]),
        ],
        initializer=[k_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 22)],
        ir_version=10,
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="TopK"):
        converter.to_coreai()


def test_topk_emits_single_sort():
    # Values must come from a gather on the argsorted indices, not a second
    # full sort of the same tensor (two O(n log n) kernels for one TopK).
    x = np.zeros((3, 5), dtype=np.float32)
    model = single_op_model(
        "TopK",
        {"x": x},
        n_outputs=2,
        attrs={"axis": -1, "largest": 1},
        initializers={"k": np.array([2], dtype=np.int64)},
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    text = str(converter.to_coreai())
    assert "coreai.sort" not in text
    assert text.count("coreai.argsort") == 1


async def test_topk_k_equals_dim():
    # k == dim should be accepted (selects all elements).
    rng = np.random.default_rng(_seed("topk-k-eq-dim"))
    x = rng.permutation(15).astype(np.float32).reshape(3, 5)
    model = single_op_model(
        "TopK",
        {"x": x},
        n_outputs=2,
        attrs={"axis": -1, "largest": 1},
        initializers={"k": np.array([5], dtype=np.int64)},  # k == dim
    )
    await assert_parity(model, {"x": x})


# ---------------------------------------------------------------------------
# Sum  (variadic elementwise add)
# ---------------------------------------------------------------------------


async def test_sum_variadic_three_inputs():
    rng = np.random.default_rng(_seed("sum-variadic"))
    a = rng.random((2, 3)).astype(np.float32)
    b = rng.random((2, 3)).astype(np.float32)
    c = rng.random((2, 3)).astype(np.float32)
    model = single_op_model("Sum", {"a": a, "b": b, "c": c})
    await assert_parity(model, {"a": a, "b": b, "c": c})
