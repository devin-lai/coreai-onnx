# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for numerical verification against onnxruntime."""

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx import (
    ModelValidationError,
    generate_inputs,
    validate_onnxruntime,
    verify,
)

from .helpers import coreai_runtime_test, single_op_model


def _sigmoid_model() -> onnx.ModelProto:
    return single_op_model("Sigmoid", {"x": np.zeros((2, 3), dtype=np.float32)})


def _dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "dyn", [x], [y])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )


# ---------------------------------------------------------------------------
# validate_onnxruntime: the pre-conversion ONNX Runtime gate
# ---------------------------------------------------------------------------


def test_validate_onnxruntime_good_model_returns_outputs():
    model = _sigmoid_model()
    out = validate_onnxruntime(model, seed=0)
    assert list(out) == ["out0"]
    assert out["out0"].shape == (2, 3)
    # the returned reference matches a hand-computed sigmoid of the seeded feed
    x = generate_inputs(model, seed=0)["x"]
    np.testing.assert_allclose(out["out0"], 1.0 / (1.0 + np.exp(-x)), rtol=1e-6)


def test_validate_onnxruntime_raises_on_runtime_failure():
    """A model that passes the static onnx.checker but errors when ONNX Runtime
    executes it (here: an out-of-range Gather index) is reported as a
    ModelValidationError, not allowed through to conversion."""
    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [2, 3])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [1])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Gather", ["data", "idx"], ["out"], axis=0)
    graph = helper.make_graph([node], "g", [data, idx], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    onnx.checker.check_model(model)  # structurally valid

    bad_inputs = {
        "data": np.zeros((2, 3), dtype=np.float32),
        "idx": np.array([5], dtype=np.int64),  # out of range for axis size 2
    }
    with pytest.raises(ModelValidationError, match="ONNX Runtime"):
        validate_onnxruntime(model, inputs=bad_inputs)


def test_validate_onnxruntime_reference_is_fusion_free():
    """The reference run must match ONNX Runtime with graph fusions disabled.

    ONNX Runtime's EXTENDED-level com.microsoft.FusedConv kernel miscomputes
    non-depthwise grouped convolution (1 < group < C) on macOS arm64 in ORT
    1.26.0, turning ResNeXt/RegNet-class references into garbage (and BASIC's
    constant folding corrupts rf_detr's box head, so reference sessions run
    with graph optimizations disabled entirely)."""
    import onnxruntime as ort

    rng = np.random.RandomState(0)
    c, hw, k, group = 128, 56, 3, 2
    w = (rng.randn(c, c // group, k, k) * 0.05).astype(np.float32)
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, hw, hw])
    y_info = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c, hw, hw])
    nodes = [
        helper.make_node(
            "Conv",
            ["x", "w"],
            ["c0"],
            group=group,
            kernel_shape=[k, k],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["c0"], ["y"]),
    ]
    graph = helper.make_graph(
        [*nodes],
        "g",
        [x_info],
        [y_info],
        [helper.make_tensor("w", TensorProto.FLOAT, w.shape, w.flatten())],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )

    feed = generate_inputs(model, seed=0)
    got = validate_onnxruntime(model, inputs=feed)["y"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    unfused = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    ).run(None, feed)[0]
    np.testing.assert_allclose(got, unfused, rtol=1e-5, atol=1e-5)


def test_validate_onnxruntime_reference_survives_sparse_conv_weights():
    """ONNX Runtime 1.24-1.26 on Apple Silicon miscomputes Conv through Arm
    KleidiAI SGEMM kernels when weights are highly sparse (~84% exact zeros,
    as in pruned models like stereonet's refiners) - the whole output is
    wrong by O(signal). The reference runner must disable KleidiAI
    ('mlas.disable_kleidiai') so the parity reference matches a float64
    ground truth."""
    rng = np.random.default_rng(5)
    c, hw = 8, 16

    def sparse_weight():
        w = (rng.standard_normal((c, c, 3, 3)) * 0.3).astype(np.float32)
        w[rng.random(w.shape) < 0.84] = 0.0
        return w

    w1, b1 = sparse_weight(), (rng.standard_normal(c) * 0.05).astype(np.float32)
    w2, b2 = sparse_weight(), (rng.standard_normal(c) * 0.05).astype(np.float32)

    def conv_node(i, o, w, b):
        return helper.make_node(
            "Conv", [i, w, b], [o], kernel_shape=[3, 3], pads=[1, 1, 1, 1]
        )

    graph = helper.make_graph(
        [
            conv_node("x", "c1", "w1", "b1"),
            helper.make_node("LeakyRelu", ["c1"], ["a1"], alpha=0.2),
            conv_node("a1", "y", "w2", "b2"),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, c, hw, hw])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, c, hw, hw])],
        [
            onnx.numpy_helper.from_array(a, n)
            for n, a in {"w1": w1, "b1": b1, "w2": w2, "b2": b2}.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )

    x = rng.standard_normal((1, c, hw, hw)).astype(np.float32)
    got = validate_onnxruntime(model, inputs={"x": x})["y"]

    def conv3x3(t, w, b):  # float64 ground truth, 3x3 / pad 1 / stride 1
        _ch, h, wd = t.shape
        tp = np.pad(t, ((0, 0), (1, 1), (1, 1)))
        out = np.zeros((w.shape[0], h, wd))
        for di in range(3):
            for dj in range(3):
                out += np.einsum(
                    "oc,chw->ohw", w[:, :, di, dj], tp[:, di : di + h, dj : dj + wd]
                )
        return out + b[:, None, None]

    c1 = conv3x3(x[0].astype(np.float64), w1.astype(np.float64), b1.astype(np.float64))
    a1 = np.where(c1 > 0, c1, 0.2 * c1)
    ref = conv3x3(a1, w2.astype(np.float64), b2.astype(np.float64))
    np.testing.assert_allclose(got[0], ref, rtol=1e-4, atol=1e-4)


def test_generate_inputs_deterministic():
    model = _sigmoid_model()
    a = generate_inputs(model, seed=0)
    b = generate_inputs(model, seed=0)
    assert list(a) == ["x"]
    assert a["x"].shape == (2, 3)
    assert a["x"].dtype == np.float32
    np.testing.assert_array_equal(a["x"], b["x"])


def test_generate_inputs_dynamic_dim_default():
    model = _dynamic_batch_model()
    inputs = generate_inputs(model)
    assert inputs["x"].shape == (2, 3)
    inputs = generate_inputs(model, dynamic_dim_size=5)
    assert inputs["x"].shape == (5, 3)


def _save_aimodel(model: onnx.ModelProto, path):
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    program = converter.to_coreai()
    program.save_asset(path)


@coreai_runtime_test
async def test_verify_roundtrip(tmp_path):
    model = _sigmoid_model()
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    report = await verify(model, asset_path, seed=0)
    assert report.passed
    out = report.outputs[0]
    assert out.name == "out0"
    assert out.passed
    assert out.max_abs_error < 1e-4
    assert out.psnr > 60 or math.isinf(out.psnr)


@coreai_runtime_test
async def test_verify_isolated_execution_matches_in_process(tmp_path):
    """isolate_execution runs the native step in a child process; for a good
    model the result must be identical to running it in-process."""
    model = _sigmoid_model()
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    in_process = await verify(model, asset_path, seed=0, isolate_execution=False)
    isolated = await verify(model, asset_path, seed=0, isolate_execution=True)
    assert isolated.passed == in_process.passed
    assert isolated.outputs[0].max_abs_error == in_process.outputs[0].max_abs_error
    assert isolated.outputs[0].psnr == in_process.outputs[0].psnr


@coreai_runtime_test
async def test_verify_explicit_inputs(tmp_path):
    model = _sigmoid_model()
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    rng = np.random.default_rng(7)
    inputs = {"x": rng.standard_normal((2, 3)).astype(np.float32)}
    report = await verify(model, asset_path, inputs=inputs)
    assert report.passed
    assert report.outputs[0].max_abs_error < 1e-4


@coreai_runtime_test
async def test_verify_multi_output(tmp_path):
    model = single_op_model(
        "Split",
        {"x": np.arange(8, dtype=np.float32)},
        n_outputs=2,
        attrs={"axis": 0, "num_outputs": 2},
    )
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    report = await verify(model, asset_path, seed=0)
    assert report.passed
    assert len(report.outputs) == 2
    assert [o.name for o in report.outputs] == ["out0", "out1"]
    assert all(o.passed for o in report.outputs)


# ---------------------------------------------------------------------------
# dtype-dependent default tolerances (float16 vs float32)
# ---------------------------------------------------------------------------


def test_compare_f16_default_tolerance():
    """With no explicit tolerances, f16 outputs get f16-scaled defaults while
    f32 outputs keep the strict ones - the same rel error flips the outcome."""
    from coreai_onnx._verify import _compare

    e16 = (np.arange(1, 65, dtype=np.float32) / 7).astype(np.float16)
    g16 = (e16.astype(np.float32) * (1 + 5e-3)).astype(np.float16)
    assert _compare("y", e16, g16, None, None).passed

    e32 = e16.astype(np.float32)
    g32 = e32 * (1 + 5e-3)
    assert not _compare("y", e32, g32, None, None).passed


def test_compare_explicit_tolerance_overrides_dtype_default():
    from coreai_onnx._verify import _compare

    e = np.array([1.0, 2.0], dtype=np.float16)
    g = (e.astype(np.float32) * 1.005).astype(np.float16)
    assert not _compare("y", e, g, 1e-4, 1e-6).passed
    assert _compare("y", e, g, 1e-2, 1e-3).passed


@coreai_runtime_test
async def test_verify_f16_model_with_default_tolerances(tmp_path):
    model = single_op_model("Floor", {"x": np.zeros((2, 3), dtype=np.float16)})
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    report = await verify(model, asset_path, seed=0)
    assert report.passed


# ---------------------------------------------------------------------------
# non-finite reference outputs (models that NaN on the random probe input,
# e.g. efficientvit's unguarded 0/0 linear-attention division)
# ---------------------------------------------------------------------------


def test_compare_matching_nan_positions_pass():
    from coreai_onnx._verify import _compare

    e = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    g = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    rep = _compare("y", e, g, None, None)
    assert rep.passed
    assert rep.expected_nonfinite == 1


def test_compare_nan_vs_finite_mismatch_fails():
    from coreai_onnx._verify import _compare

    e = np.array([1.0, np.nan], dtype=np.float32)
    g = np.array([1.0, 2.0], dtype=np.float32)
    assert not _compare("y", e, g, None, None).passed
    # and the converted model NaN-ing where the reference is finite also fails
    assert not _compare("y", g, e, None, None).passed
    assert _compare("y", g, e, None, None).expected_nonfinite == 0


def test_compare_metrics_skip_nonfinite_positions():
    """Error metrics describe the finite-overlap region instead of being
    NaN-poisoned by positions the pass/fail logic already accounts for."""
    from coreai_onnx._verify import _compare

    e = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    g = np.array([1.0, np.nan, 2.001], dtype=np.float32)
    rep = _compare("y", e, g, None, None)
    assert not math.isnan(rep.max_abs_error)
    assert rep.max_abs_error == pytest.approx(1e-3, rel=0.2)
    assert not math.isnan(rep.psnr)


def test_compare_all_nan_reference_passes_with_count():
    from coreai_onnx._verify import _compare

    e = np.full((4,), np.nan, dtype=np.float32)
    rep = _compare("y", e, e.copy(), None, None)
    assert rep.passed
    assert rep.expected_nonfinite == 4
    assert rep.max_abs_error == 0.0


# ---------------------------------------------------------------------------
# min_psnr: PSNR-based acceptance for large-magnitude accumulation noise
# ---------------------------------------------------------------------------


def test_compare_min_psnr_accepts_high_psnr_output():
    from coreai_onnx._verify import _compare

    e = (np.arange(1, 1001, dtype=np.float64) * 1e6).astype(np.float32)
    g = e * (1 + 2e-3)  # uniform 2e-3 relative error: fails rtol=1e-3
    assert not _compare("y", e, g, None, None).passed
    rep = _compare("y", e, g, None, None, min_psnr=40.0)
    assert rep.passed
    assert rep.psnr > 40.0


def test_compare_min_psnr_still_fails_low_psnr_output():
    from coreai_onnx._verify import _compare

    e = np.arange(1, 101, dtype=np.float32)
    g = e * 1.5
    assert not _compare("y", e, g, None, None, min_psnr=40.0).passed


def test_compare_min_psnr_does_not_mask_nan_mismatch():
    from coreai_onnx._verify import _compare

    e = np.array([1e6, 2e6], dtype=np.float32)
    g = np.array([1e6, np.nan], dtype=np.float32)
    assert not _compare("y", e, g, None, None, min_psnr=0.0).passed


# ---------------------------------------------------------------------------
# compute_unit selection
# ---------------------------------------------------------------------------


@coreai_runtime_test
async def test_verify_compute_unit_cpu_only(tmp_path):
    model = _sigmoid_model()
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    report = await verify(model, asset_path, seed=0, compute_unit="cpu_only")
    assert report.passed


async def test_verify_unknown_compute_unit_rejected(tmp_path):
    model = _sigmoid_model()
    with pytest.raises(Exception, match="unknown compute_unit 'tpu'"):
        await verify(model, tmp_path / "missing.aimodel", compute_unit="tpu")


# ---------------------------------------------------------------------------
# entrypoint / output_names parameters
# ---------------------------------------------------------------------------


@coreai_runtime_test
async def test_verify_custom_entrypoint(tmp_path):
    model = _sigmoid_model()
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model, entrypoint_name="encoder")
    converter.to_coreai().save_asset(tmp_path / "m.aimodel")
    report = await verify(model, tmp_path / "m.aimodel", entrypoint="encoder")
    assert report.passed


@coreai_runtime_test
async def test_verify_renamed_outputs(tmp_path):
    model = _sigmoid_model()
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model, output_names=["renamed"])
    converter.to_coreai().save_asset(tmp_path / "m.aimodel")
    report = await verify(model, tmp_path / "m.aimodel", output_names=["renamed"])
    assert report.passed
    assert report.outputs[0].name == "renamed"


@coreai_runtime_test
async def test_verify_uint64_input_roundtrip(tmp_path):
    """The feed must be narrowed uint64->uint32 to match the asset's narrowed
    input dtype; otherwise the runtime rejects the uint64 feed."""
    x = helper.make_tensor_value_info("x", TensorProto.UINT64, [4])
    y = helper.make_tensor_value_info("y", TensorProto.UINT64, [4])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "g", [x], [y]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    model = onnx.shape_inference.infer_shapes(model)
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    report = await verify(model, asset_path, seed=0)
    assert report.passed


@coreai_runtime_test
async def test_verify_flags_lossy_int64_overflow(tmp_path):
    """A true int64 output exceeding int32 cannot be represented by the int32
    .aimodel; verify must compare against the true onnxruntime output and report
    the gap, not narrow the reference into a matching (false) pass."""
    x = helper.make_tensor_value_info("x", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "x"], ["y"])], "g", [x], [y]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    model = onnx.shape_inference.infer_shapes(model)
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)
    # 100000**2 = 1e10 overflows int32; the asset's int32 Mul cannot match it.
    inputs = {"x": np.array([100000, 3], dtype=np.int64)}
    report = await verify(model, asset_path, inputs=inputs)
    assert not report.passed
    assert report.outputs[0].max_abs_error > 1.0


@coreai_runtime_test
async def test_verify_reuses_supplied_expected_outputs(tmp_path, monkeypatch):
    """When ``expected`` is supplied, verify must not re-run ONNX Runtime - the
    convert pipeline relies on this to run ORT exactly once (validate + verify)."""
    import coreai_onnx._verify as _v

    model = _sigmoid_model()
    asset_path = tmp_path / "m.aimodel"
    _save_aimodel(model, asset_path)

    inputs = generate_inputs(model, seed=0)
    expected = {"out0": (1.0 / (1.0 + np.exp(-inputs["x"]))).astype(np.float32)}

    def _boom(*_a, **_k):
        raise AssertionError(
            "_run_onnxruntime must not be called when expected= is given"
        )

    monkeypatch.setattr(_v, "_run_onnxruntime", _boom)
    report = await verify(model, asset_path, inputs=inputs, expected=expected)
    assert report.passed
    assert report.outputs[0].max_abs_error < 1e-4
