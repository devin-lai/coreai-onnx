# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_onnx._cli - driven via main(argv) directly."""

import numpy as np
import onnx

from coreai_onnx._cli import main
from tests.helpers import (
    COREAI_CONVERSION_MARKS,
    coreai_runtime_test,
    det_model_file,
    relu_model_file,
)

pytestmark = [*COREAI_CONVERSION_MARKS]

# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_supported(tmp_path, capsys):
    rc = main(["inspect", relu_model_file(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Relu" in out
    assert any(word in out.lower() for word in ["yes", "convertible", "true"])


def test_inspect_unsupported(tmp_path, capsys):
    rc = main(["inspect", det_model_file(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Det" in out


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


@coreai_runtime_test
def test_convert_command(tmp_path, capsys):
    model_path = relu_model_file(tmp_path)
    out_path = str(tmp_path / "out.aimodel")
    rc = main(["convert", model_path, "-o", out_path])
    assert rc == 0
    from pathlib import Path

    assert Path(out_path).exists()


def test_convert_unsupported_returns_1(tmp_path, capsys):
    model_path = det_model_file(tmp_path)
    out_path = str(tmp_path / "out.aimodel")
    rc = main(["convert", model_path, "-o", out_path])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert rc == 1
    # Should print a helpful message, not a raw traceback
    assert "Det" in out or "unsupported" in out.lower() or "lowering" in out.lower()


def test_convert_missing_file_clean_error(tmp_path, capsys):
    rc = main(
        [
            "convert",
            str(tmp_path / "missing.onnx"),
            "-o",
            str(tmp_path / "out.aimodel"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_inspect_missing_file_clean_error(tmp_path, capsys):
    rc = main(["inspect", str(tmp_path / "missing.onnx")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_convert_corrupt_model_clean_error(tmp_path, capsys):
    bad = tmp_path / "corrupt.onnx"
    bad.write_bytes(b"\x00not a protobuf at all")
    rc = main(["convert", str(bad), "-o", str(tmp_path / "out.aimodel")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "Error" in out


def test_convert_overflow_initializer_clean_error(tmp_path, capsys):
    """A checker-valid model whose int64 initializer exceeds int32 must produce
    a clean diagnostic and exit 1, not a raw OverflowError traceback."""
    from onnx import TensorProto, helper, numpy_helper

    big = numpy_helper.from_array(np.array([1, 2**40], dtype=np.int64), name="big")
    node = helper.make_node("Add", ["X", "big"], ["Y"])
    graph = helper.make_graph(
        [node],
        "g",
        inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, [2])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, [2])],
        initializer=[big],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    path = tmp_path / "overflow.onnx"
    onnx.save(model, str(path))
    rc = main(["convert", str(path), "-o", str(tmp_path / "out.aimodel")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "int32" in out


@coreai_runtime_test
def test_convert_loads_model_once(tmp_path, monkeypatch):
    """The success banner must not reload the .onnx from disk a second time."""
    model_path = relu_model_file(tmp_path)
    calls = []
    real_load = onnx.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr("coreai_onnx._service.onnx.load", counting_load)
    rc = main(["convert", model_path, "-o", str(tmp_path / "out.aimodel")])
    assert rc == 0
    assert len(calls) == 1


@coreai_runtime_test
def test_no_optimize_flag(tmp_path, capsys):
    model_path = relu_model_file(tmp_path)
    out_path = str(tmp_path / "out_noopt.aimodel")
    rc = main(["convert", model_path, "-o", out_path, "--no-optimize"])
    assert rc == 0
    from pathlib import Path

    assert Path(out_path).exists()


# ---------------------------------------------------------------------------
# convert pipeline: ONNX Runtime validation gate + precision check
# ---------------------------------------------------------------------------


def _gather_oob_model_file(tmp_path) -> str:
    """A structurally valid but ORT-unrunnable model (out-of-range Gather): the
    seeded index feed lands outside a size-1 axis, so ONNX Runtime errors."""
    from onnx import TensorProto, helper

    data = helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 3])
    idx = helper.make_tensor_value_info("idx", TensorProto.INT64, [4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [4, 3])
    node = helper.make_node("Gather", ["data", "idx"], ["out"], axis=0)
    graph = helper.make_graph([node], "g", [data, idx], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    path = tmp_path / "gather_oob.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_convert_validation_failure_returns_1(tmp_path, capsys, monkeypatch):
    """A model that fails the ONNX Runtime gate is rejected before conversion:
    exit 1, a clean message, and no .aimodel written."""
    import coreai_onnx
    from coreai_onnx import ModelValidationError

    def _boom(*_a, **_k):
        raise ModelValidationError(
            "the input ONNX model failed to run on ONNX Runtime: boom"
        )

    monkeypatch.setattr(coreai_onnx, "validate_onnxruntime", _boom)
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(["convert", model_path, "-o", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "ONNX Runtime" in out
    assert not out_path.exists()


@coreai_runtime_test
def test_convert_no_validate_bypasses_gate(tmp_path, capsys):
    """--no-validate lets an ORT-invalid-but-convertible model through (the gate
    would otherwise reject it at exit 1). --no-verify too, since the same random
    indices would break the precision run."""
    model_path = _gather_oob_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(
        ["convert", model_path, "-o", str(out_path), "--no-validate", "--no-verify"]
    )
    assert rc == 0
    assert out_path.exists()


@coreai_runtime_test
def test_convert_runs_precision_check(tmp_path, capsys):
    """On macOS with onnxruntime, convert auto-prints the precision comparison."""
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(["convert", model_path, "-o", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_path.exists()
    assert "PSNR" in out
    assert "passed" in out.lower()


@coreai_runtime_test
def test_convert_no_verify_skips_precision_check(tmp_path, capsys):
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(["convert", model_path, "-o", str(out_path), "--no-verify"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_path.exists()
    assert "PSNR" not in out


@coreai_runtime_test
def test_convert_skips_onnxruntime_steps_when_absent(tmp_path, capsys, monkeypatch):
    """Without onnxruntime, both ORT-dependent steps are skipped with a note and
    conversion still succeeds (exit 0)."""
    import coreai_onnx._service as service

    monkeypatch.setattr(service, "_onnxruntime_available", lambda: False)
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(["convert", model_path, "-o", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_path.exists()
    assert "not installed" in out
    assert "PSNR" not in out


@coreai_runtime_test
def test_convert_precision_failure_returns_3(tmp_path, capsys, monkeypatch):
    """When the precision check fails tolerance, the .aimodel is still written
    but convert exits 3 (distinct from a conversion failure at exit 1)."""
    import coreai_onnx
    from coreai_onnx import OutputReport, VerifyReport

    async def _fake_verify(*_a, **_k):
        return VerifyReport(
            outputs=[OutputReport("y", 1.0, 0.5, 12.0, passed=False)], passed=False
        )

    monkeypatch.setattr(coreai_onnx, "verify", _fake_verify)
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc = main(["convert", model_path, "-o", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 3
    assert out_path.exists()
    assert "failed" in out.lower()


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


@coreai_runtime_test
def test_verify_command(tmp_path, capsys):
    # First convert to produce the .aimodel
    model_path = relu_model_file(tmp_path)
    aimodel_path = str(tmp_path / "relu.aimodel")
    rc = main(["convert", model_path, "-o", aimodel_path])
    assert rc == 0

    # Now verify
    rc = main(["verify", model_path, aimodel_path])
    out = capsys.readouterr().out
    assert rc == 0
    assert (
        "pass" in out.lower() or "✓" in out or "PASS" in out or "passed" in out.lower()
    )


@coreai_runtime_test
def test_verify_name_flag(tmp_path, capsys):
    """convert --name encoder followed by verify --name encoder must pass."""
    model_path = relu_model_file(tmp_path)
    aimodel_path = str(tmp_path / "relu_named.aimodel")
    rc = main(["convert", model_path, "-o", aimodel_path, "--name", "encoder"])
    assert rc == 0

    rc = main(["verify", model_path, aimodel_path, "--name", "encoder"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pass" in out.lower()


def test_verify_non_darwin_returns_2(tmp_path, monkeypatch, capsys):
    """On a non-macOS platform verify exits 2 with a clean message instead of
    attempting to execute the asset."""
    import coreai_onnx._service as service

    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    rc = main(["verify", str(tmp_path / "x.onnx"), str(tmp_path / "x.aimodel")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "macOS" in out


def test_verify_human_output_handles_nonfinite_metrics(tmp_path, monkeypatch, capsys):
    """Human rendering must accept the same strict-JSON-safe metric strings as
    --json output, instead of trying to format them as floats."""
    import coreai_onnx
    import coreai_onnx._service as service
    from coreai_onnx import OutputReport, VerifyReport

    monkeypatch.setattr(service.platform, "system", lambda: "Darwin")

    async def _nonfinite_verify(*_a, **_k):
        return VerifyReport(
            outputs=[
                OutputReport(
                    name="y",
                    max_abs_error=float("inf"),
                    max_rel_error=float("nan"),
                    psnr=float("-inf"),
                    passed=False,
                )
            ],
            passed=False,
        )

    monkeypatch.setattr(coreai_onnx, "verify", _nonfinite_verify)
    rc = main(["verify", relu_model_file(tmp_path), str(tmp_path / "x.aimodel")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "inf" in out
    assert "nan" in out
    assert "Verification failed" in out


# ---------------------------------------------------------------------------
# entry-point importability
# ---------------------------------------------------------------------------


def test_import_main():
    from coreai_onnx._cli import main as _main

    assert callable(_main)
