# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""JSON-mode contract tests: one envelope on stdout, stable codes, frozen
exit codes. See docs/cli.md for the public contract."""

import json

import numpy as np
import onnx

from coreai_onnx._cli import main
from tests.helpers import (
    COREAI_CONVERSION_MARKS,
    ENVELOPE_KEYS,
    coreai_runtime_test,
    det_model_file,
    relu_model_file,
    single_op_model,
)

pytestmark = [*COREAI_CONVERSION_MARKS]


def _reject_nonfinite(token: str) -> float:
    raise AssertionError(f"non-finite JSON token leaked into output: {token}")


def _run_json(argv, capsys):
    """Run main(argv), assert stdout is exactly one strict-JSON document, and
    return (exit_code, envelope)."""
    rc = main(argv)
    out = capsys.readouterr().out
    # parse_constant rejects bare Infinity/-Infinity/NaN tokens, which
    # json.dumps would emit for float("inf") — the envelope must serialize
    # those as strings (e.g. psnr: "inf") to stay valid JSON.
    env = json.loads(out, parse_constant=_reject_nonfinite)
    assert set(env) == ENVELOPE_KEYS
    return rc, env


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def test_inspect_json_supported(tmp_path, capsys):
    rc, env = _run_json(["inspect", relu_model_file(tmp_path), "--json"], capsys)
    assert rc == 0
    assert env["schema_version"] == 1
    assert env["command"] == "inspect"
    assert env["status"] == "ok"
    assert env["error"] is None
    r = env["result"]
    assert r["convertible"] is True
    assert r["total_nodes"] == 1
    assert r["unsupported"] == []
    assert {"op": "Relu", "count": 1, "supported": True} in r["ops"]


def test_inspect_json_unsupported_exits_1_status_ok(tmp_path, capsys):
    # 'inspect' succeeded at analyzing; convertible=False is a result, not an
    # error. The exit code still signals non-convertibility per the contract.
    rc, env = _run_json(["inspect", det_model_file(tmp_path), "--json"], capsys)
    assert rc == 1
    assert env["status"] == "ok"
    assert env["result"]["convertible"] is False
    assert "Det" in env["result"]["unsupported"]


def test_json_flag_accepted_before_subcommand(tmp_path, capsys):
    rc, env = _run_json(["--json", "inspect", relu_model_file(tmp_path)], capsys)
    assert rc == 0
    assert env["command"] == "inspect"


def _sigmoid_add_model_file(tmp_path) -> str:
    """Two-op model: Sigmoid -> Add (graph order Sigmoid, Add; sorted Add, Sigmoid).

    Sigmoid(x) -> s; Add(s, x) -> out. Both ops are supported, so convertible=True.
    """
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [4])
    out = onnx.helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [4])
    sigmoid_node = onnx.helper.make_node("Sigmoid", inputs=["x"], outputs=["s"])
    add_node = onnx.helper.make_node("Add", inputs=["s", "x"], outputs=["out"])
    graph = onnx.helper.make_graph([sigmoid_node, add_node], "test_sorted", [x], [out])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 22)],
        ir_version=10,
    )
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    path = tmp_path / "sigmoid_add.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_inspect_json_ops_sorted(tmp_path, capsys):
    # Graph order: Sigmoid, Add. Sorted order: Add, Sigmoid.
    # Verifies the response list is sorted alphabetically, not graph-order.
    _, env = _run_json(["inspect", _sigmoid_add_model_file(tmp_path), "--json"], capsys)
    r = env["result"]
    assert r["convertible"] is True
    assert r["total_nodes"] == 2
    ops = [e["op"] for e in r["ops"]]
    assert ops == ["Add", "Sigmoid"], f"expected sorted order, got {ops}"
    counts = {e["op"]: e["count"] for e in r["ops"]}
    assert counts == {"Add": 1, "Sigmoid": 1}


# ---------------------------------------------------------------------------
# structured errors (exception paths reaching main)
# ---------------------------------------------------------------------------


def test_convert_json_unsupported_ops(tmp_path, capsys):
    rc, env = _run_json(
        [
            "convert",
            det_model_file(tmp_path),
            "-o",
            str(tmp_path / "out.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["status"] == "error"
    assert env["result"] is None
    err = env["error"]
    assert err["code"] == "unsupported_ops"
    assert "Det" in str(err["details"]["missing"])
    assert err["hint"]  # remediation present


def test_convert_json_missing_file_is_io_error(tmp_path, capsys):
    rc, env = _run_json(
        [
            "convert",
            str(tmp_path / "missing.onnx"),
            "-o",
            str(tmp_path / "out.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["error"]["code"] == "io_error"


def test_convert_json_corrupt_model_is_invalid_model_file(tmp_path, capsys):
    bad = tmp_path / "corrupt.onnx"
    bad.write_bytes(b"\x00\x01not a protobuf")
    rc, env = _run_json(
        ["convert", str(bad), "-o", str(tmp_path / "out.aimodel"), "--json"], capsys
    )
    assert rc == 1
    assert env["error"]["code"] == "invalid_model_file"


def test_convert_human_ort_note_gating(tmp_path, capsys, monkeypatch):
    """Human-mode ORT note must NOT appear for io_error (missing file), but
    MUST appear for unsupported-ops (model loaded, conversion failed)."""
    import coreai_onnx._service as service

    monkeypatch.setattr(service, "_onnxruntime_available", lambda: False)

    # Case 1: missing file → io_error; note must NOT be printed.
    rc = main(
        [
            "convert",
            str(tmp_path / "missing.onnx"),
            "-o",
            str(tmp_path / "out1.aimodel"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "not installed" not in out, "ORT note must not appear for io_error"
    assert "Error" in out

    # Case 2: model loads but has unsupported ops → note MUST appear before error.
    rc = main(
        [
            "convert",
            det_model_file(tmp_path),
            "-o",
            str(tmp_path / "out2.aimodel"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "not installed" in out, "ORT note must appear for unsupported-ops error"
    note_pos = out.index("not installed")
    error_pos = out.index("Error")
    assert note_pos < error_pos, "ORT note must precede the Error line"


def test_convert_json_ort_note_gating(tmp_path, capsys, monkeypatch):
    """JSON mode: unsupported-op convert with ORT absent → warnings contains
    onnxruntime_missing; missing-file case → warnings is empty."""
    import coreai_onnx._service as service

    monkeypatch.setattr(service, "_onnxruntime_available", lambda: False)

    # Case 1: missing file → io_error; warnings must be empty.
    rc, env = _run_json(
        [
            "convert",
            str(tmp_path / "missing.onnx"),
            "-o",
            str(tmp_path / "out1.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["error"]["code"] == "io_error"
    assert env["warnings"] == [], "ORT note must NOT appear for io_error"

    # Case 2: model loads but has unsupported ops → onnxruntime_missing warning.
    rc, env = _run_json(
        [
            "convert",
            det_model_file(tmp_path),
            "-o",
            str(tmp_path / "out2.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["error"]["code"] == "unsupported_ops"
    codes = [w["code"] for w in env["warnings"]]
    assert "onnxruntime_missing" in codes, f"Expected onnxruntime_missing in {codes}"


def test_convert_json_validation_failure(tmp_path, capsys, monkeypatch):
    import coreai_onnx

    def _boom(*_a, **_k):
        raise coreai_onnx.ModelValidationError("model failed to run on ONNX Runtime")

    monkeypatch.setattr(coreai_onnx, "validate_onnxruntime", _boom)
    rc, env = _run_json(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(tmp_path / "out.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["error"]["code"] == "model_validation_failed"


# ---------------------------------------------------------------------------
# convert: success, warnings, precision
# ---------------------------------------------------------------------------


@coreai_runtime_test
def test_convert_json_success(tmp_path, capsys):
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"], capsys
    )
    assert rc == 0
    assert env["status"] == "ok"
    r = env["result"]
    assert r["output_path"] == str(out_path)
    assert r["total_nodes"] == 1
    assert r["optimized"] is True
    assert out_path.exists()
    # On the runtime, the precision check ran and passed.
    assert r["precision"]["passed"] is True
    assert r["precision"]["min_psnr"] is None
    assert r["precision"]["compute_unit"] is None
    assert all(
        set(o)
        == {
            "name",
            "max_abs_error",
            "max_rel_error",
            "psnr",
            "passed",
            "expected_nonfinite",
        }
        for o in r["precision"]["outputs"]
    )


@coreai_runtime_test
def test_convert_json_reference_nonfinite_warning(tmp_path, capsys):
    """A model that produces NaN on the probe input (Sqrt of a negative) gets
    a reference_nonfinite warning; matching NaN positions still pass."""
    model = single_op_model("Sqrt", {"x": np.zeros((8,), dtype=np.float32)})
    path = tmp_path / "sqrt.onnx"
    onnx.save(model, str(path))
    out_path = tmp_path / "out.aimodel"
    # seed 0's standard normal draw contains negatives, so the reference
    # output contains NaN at those positions.
    rc, env = _run_json(["convert", str(path), "-o", str(out_path), "--json"], capsys)
    assert rc == 0, env["error"]
    codes = [w["code"] for w in env["warnings"]]
    assert "reference_nonfinite" in codes
    outs = env["result"]["precision"]["outputs"]
    assert outs[0]["expected_nonfinite"] > 0
    assert outs[0]["passed"] is True


@coreai_runtime_test
def test_convert_json_min_psnr_recorded_and_passed_through(tmp_path, capsys):
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(out_path),
            "--min-psnr",
            "40",
            "--json",
        ],
        capsys,
    )
    assert rc == 0
    assert env["result"]["precision"]["min_psnr"] == 40.0


@coreai_runtime_test
def test_convert_json_compute_unit_cpu_only(tmp_path, capsys):
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(out_path),
            "--compute-unit",
            "cpu_only",
            "--json",
        ],
        capsys,
    )
    assert rc == 0
    assert env["result"]["precision"]["compute_unit"] == "cpu_only"
    assert env["result"]["precision"]["passed"] is True


@coreai_runtime_test
def test_convert_json_overwrites_existing_aimodel(tmp_path, capsys):
    """Re-converting to the same output path must succeed: the previous
    .aimodel bundle is replaced (save_asset itself cannot overwrite)."""
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "out.aimodel"
    rc1, _ = _run_json(["convert", model_path, "-o", str(out_path), "--json"], capsys)
    assert rc1 == 0
    rc2, env = _run_json(["convert", model_path, "-o", str(out_path), "--json"], capsys)
    assert rc2 == 0, env["error"]
    assert env["status"] == "ok"
    assert out_path.exists()


@coreai_runtime_test
def test_convert_stamps_provenance_metadata(tmp_path, capsys):
    from coreai.authoring import AIModelAsset

    out_path = tmp_path / "out.aimodel"
    rc, _ = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"], capsys
    )
    assert rc == 0

    metadata = AIModelAsset.load(out_path).metadata
    assert metadata.author == "coreai-onnx contributors"
    assert metadata.license == "BSD-3-Clause"
    assert metadata.model_description == (
        "Converted with coreai-onnx: https://github.com/devin-lai/coreai-onnx"
    )
    assert metadata.creator_defined_metadata["coreai_onnx.repository"] == (
        "https://github.com/devin-lai/coreai-onnx"
    )
    assert metadata.creator_defined_metadata["coreai_onnx.license"] == "BSD-3-Clause"


def test_convert_json_refuses_to_replace_non_aimodel_directory(tmp_path, capsys):
    """An existing output path that is NOT an .aimodel bundle must not be
    deleted; the convert fails with a clear io_error instead."""
    model_path = relu_model_file(tmp_path)
    out_path = tmp_path / "precious"
    out_path.mkdir()
    (out_path / "keep.txt").write_text("do not delete")
    rc, env = _run_json(["convert", model_path, "-o", str(out_path), "--json"], capsys)
    assert rc == 1
    assert env["error"]["code"] == "io_error"
    assert "not an .aimodel bundle" in env["error"]["message"]
    assert (out_path / "keep.txt").exists()


def test_convert_invalid_compute_unit_is_usage_error(tmp_path, capsys):
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "convert",
                relu_model_file(tmp_path),
                "-o",
                str(tmp_path / "o.aimodel"),
                "--compute-unit",
                "tpu",
            ]
        )
    assert exc_info.value.code == 2


def test_convert_json_ort_missing_warning(tmp_path, capsys, monkeypatch):
    from coreai_onnx import _service

    monkeypatch.setattr(_service, "_onnxruntime_available", lambda: False)
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"], capsys
    )
    assert rc == 0
    assert env["status"] == "ok"
    assert env["result"]["validated"] is False
    assert env["result"]["precision"] is None
    codes = [w["code"] for w in env["warnings"]]
    assert "onnxruntime_missing" in codes


@coreai_runtime_test
def test_convert_json_precision_failure_exits_3(tmp_path, capsys, monkeypatch):
    import coreai_onnx
    from coreai_onnx import OutputReport, VerifyReport

    async def _fake_verify(*_a, **_k):
        return VerifyReport(
            outputs=[
                OutputReport(
                    name="y",
                    max_abs_error=1.0,
                    max_rel_error=1.0,
                    psnr=1.0,
                    passed=False,
                )
            ],
            passed=False,
        )

    monkeypatch.setattr(coreai_onnx, "verify", _fake_verify)
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"], capsys
    )
    assert rc == 3
    assert env["status"] == "error"
    assert env["error"]["code"] == "precision_check_failed"
    # Partial result is kept: the asset WAS written and precision data exists.
    assert env["result"]["precision"]["passed"] is False
    assert out_path.exists()


@coreai_runtime_test
def test_convert_json_precision_error_exits_3(tmp_path, capsys, monkeypatch):
    import coreai_onnx

    async def _broken_verify(*_a, **_k):
        raise RuntimeError("runtime exploded")

    monkeypatch.setattr(coreai_onnx, "verify", _broken_verify)
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"],
        capsys,
    )
    assert rc == 3
    assert env["error"]["code"] == "precision_check_error"
    assert env["result"] is not None  # asset written; partial result retained
    assert out_path.exists()


# ---------------------------------------------------------------------------
# convert --repair (automatic known-safe rewrites)
# ---------------------------------------------------------------------------


def _fp16_model_file(tmp_path) -> str:
    """A float16 model: the Core AI runtime cannot load it without --repair."""
    x = np.zeros((2, 3), dtype=np.float16)
    w = np.ones((3,), dtype=np.float16)
    model = single_op_model("Mul", {"x": x}, initializers={"w": w})
    path = tmp_path / "fp16.onnx"
    onnx.save(model, str(path))
    return str(path)


def test_convert_json_repair_promotes_float16(tmp_path, capsys):
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        [
            "convert",
            _fp16_model_file(tmp_path),
            "-o",
            str(out_path),
            "--repair",
            "--json",
        ],
        capsys,
    )
    assert rc == 0, env["error"]
    assert env["status"] == "ok"
    repairs = env["result"]["repairs"]
    assert [r["name"] for r in repairs] == ["promote_float16_to_float32"]
    assert repairs[0]["details"]["inputs"] == ["x"]
    assert out_path.exists()


def test_convert_json_repairs_field_empty_without_flag(tmp_path, capsys):
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(out_path),
            "--no-verify",
            "--json",
        ],
        capsys,
    )
    assert rc == 0
    assert env["result"]["repairs"] == []


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_json_non_darwin_platform_unsupported(tmp_path, capsys, monkeypatch):
    import coreai_onnx._service as _service

    monkeypatch.setattr(_service.platform, "system", lambda: "Linux")
    rc, env = _run_json(
        ["verify", relu_model_file(tmp_path), str(tmp_path / "x.aimodel"), "--json"],
        capsys,
    )
    assert rc == 2
    assert env["status"] == "error"
    assert env["error"]["code"] == "platform_unsupported"
    assert env["result"] is None


@coreai_runtime_test
def test_verify_json_pass_and_inf_psnr(tmp_path, capsys):
    # Convert first (quietly), then verify via JSON. A Relu on zeros gives an
    # exact match, so PSNR is infinite and must serialize as the string "inf".
    model_path = relu_model_file(tmp_path)
    out_path = str(tmp_path / "out.aimodel")
    assert main(["convert", model_path, "-o", out_path, "--no-verify"]) == 0
    capsys.readouterr()  # discard convert's human output

    rc, env = _run_json(["verify", model_path, out_path, "--json"], capsys)
    assert rc == 0
    assert env["status"] == "ok"
    r = env["result"]
    assert r["passed"] is True
    assert r["seed"] == 0
    assert r["rtol"] is None  # null = per-dtype default chosen inside verify()
    for o in r["outputs"]:
        assert o["psnr"] == "inf" or isinstance(o["psnr"], float)


@coreai_runtime_test
def test_verify_json_failure_exits_1_with_result(tmp_path, capsys, monkeypatch):
    import coreai_onnx
    from coreai_onnx import OutputReport, VerifyReport

    model_path = relu_model_file(tmp_path)
    out_path = str(tmp_path / "out.aimodel")
    assert main(["convert", model_path, "-o", out_path, "--no-verify"]) == 0
    capsys.readouterr()  # discard convert's human output

    async def _failing_verify(*_a, **_k):
        return VerifyReport(
            outputs=[
                OutputReport(
                    name="y",
                    max_abs_error=1.0,
                    max_rel_error=1.0,
                    psnr=1.0,
                    passed=False,
                )
            ],
            passed=False,
        )

    monkeypatch.setattr(coreai_onnx, "verify", _failing_verify)
    rc, env = _run_json(["verify", model_path, out_path, "--json"], capsys)
    assert rc == 1
    assert env["status"] == "error"
    assert env["error"]["code"] == "precision_check_failed"
    assert env["result"]["passed"] is False  # partial result retained


def test_verify_json_nonfinite_metrics_serialize_as_strings(
    tmp_path, capsys, monkeypatch
):
    # An all-zero reference with nonzero got yields psnr=-inf (_compare's
    # documented "zero signal, nonzero noise" value); NaN/Inf model outputs
    # produce non-finite error metrics. JSON has no literals for any of these —
    # the envelope must carry them as strings, like the existing psnr "inf".
    import coreai_onnx
    import coreai_onnx._service as _service
    from coreai_onnx import OutputReport, VerifyReport

    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")

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
    rc, env = _run_json(
        ["verify", relu_model_file(tmp_path), str(tmp_path / "x.aimodel"), "--json"],
        capsys,
    )
    assert rc == 1
    o = env["result"]["outputs"][0]
    assert o["max_abs_error"] == "inf"
    assert o["max_rel_error"] == "nan"
    assert o["psnr"] == "-inf"


def test_verify_json_nonfinite_tolerances_serialize_as_strings(
    tmp_path, capsys, monkeypatch
):
    # The requested --rtol/--atol/--min-psnr are echoed back in result. A user
    # may pass non-finite thresholds (e.g. --rtol inf to accept any error);
    # like the metrics, these must serialize as strings, not crash
    # json.dumps(allow_nan=False) with a bare Infinity/NaN token.
    import coreai_onnx
    import coreai_onnx._service as _service
    from coreai_onnx import OutputReport, VerifyReport

    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")

    async def _passing_verify(*_a, **_k):
        return VerifyReport(
            outputs=[
                OutputReport(
                    name="y",
                    max_abs_error=0.0,
                    max_rel_error=0.0,
                    psnr=1.0,
                    passed=True,
                )
            ],
            passed=True,
        )

    monkeypatch.setattr(coreai_onnx, "verify", _passing_verify)
    rc, env = _run_json(
        [
            "verify",
            relu_model_file(tmp_path),
            str(tmp_path / "x.aimodel"),
            "--rtol",
            "inf",
            "--atol",
            "nan",
            "--min-psnr",
            "inf",
            "--json",
        ],
        capsys,
    )
    assert rc == 0
    r = env["result"]
    assert r["rtol"] == "inf"
    assert r["atol"] == "nan"
    assert r["min_psnr"] == "inf"


# ---------------------------------------------------------------------------
# verify: missing file → io_error (not precision_check_error)
# ---------------------------------------------------------------------------


def test_verify_json_missing_file_is_io_error(tmp_path, capsys, monkeypatch):
    # The file doesn't exist so verify() raises before touching the runtime;
    # the blanket except in _run_verify must NOT swallow OSError — it should
    # re-raise so main()'s _error_from_exception classifies it as io_error.
    import coreai_onnx._service as _service

    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")
    rc, env = _run_json(
        [
            "verify",
            str(tmp_path / "missing.onnx"),
            str(tmp_path / "x.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 1
    assert env["status"] == "error"
    assert env["error"]["code"] == "io_error", (
        f"expected io_error, got {env['error']['code']!r} "
        f"(details: {env['error'].get('details')})"
    )


# ---------------------------------------------------------------------------
# convert: platform_no_runtime warning (non-Darwin skips precision check)
# ---------------------------------------------------------------------------


@coreai_runtime_test
def test_convert_json_platform_no_runtime_warning(tmp_path, capsys, monkeypatch):
    import coreai_onnx._service as _service

    monkeypatch.setattr(_service.platform, "system", lambda: "Linux")
    out_path = tmp_path / "out.aimodel"
    rc, env = _run_json(
        ["convert", relu_model_file(tmp_path), "-o", str(out_path), "--json"],
        capsys,
    )
    assert rc == 0
    assert env["status"] == "ok"
    r = env["result"]
    assert r["precision"] is None
    codes = [w["code"] for w in env["warnings"]]
    assert "platform_no_runtime" in codes, f"expected platform_no_runtime in {codes}"


# ---------------------------------------------------------------------------
# non-interactive guarantees + schema guard
# ---------------------------------------------------------------------------


def test_human_mode_no_ansi_when_piped(tmp_path, capsys):
    # capsys captures through a non-TTY pipe; rich must emit no escape codes.
    main(["inspect", relu_model_file(tmp_path)])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_json_mode_stdout_is_single_document_even_with_warnings(
    tmp_path, capsys, monkeypatch
):
    # Named guard for the "--json stdout carries exactly one JSON document"
    # contract: warnings must land inside the envelope, never as bare print()
    # prose before/after it. test_convert_json_ort_missing_warning checks the
    # warning's *content*; this pins the stdout *purity* contract by name
    # (the parse itself happens in _run_json and would fail on interleaving).
    from coreai_onnx import _service

    monkeypatch.setattr(_service, "_onnxruntime_available", lambda: False)
    rc, env = _run_json(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(tmp_path / "out.aimodel"),
            "--json",
        ],
        capsys,
    )
    assert rc == 0
    assert env["warnings"]  # the warning path was actually exercised


def test_schema_version_is_frozen(tmp_path, capsys):
    # schema_version bumps only on breaking changes. If this test fails you
    # are making a breaking change to the envelope: bump deliberately and
    # update docs/cli.md, do not just fix the assertion.
    _, env = _run_json(["inspect", relu_model_file(tmp_path), "--json"], capsys)
    assert env["schema_version"] == 1
    assert set(env) == ENVELOPE_KEYS


# ---------------------------------------------------------------------------
# schema (capability dump)
# ---------------------------------------------------------------------------


def test_schema_json_shape(capsys):
    rc, env = _run_json(["schema", "--json"], capsys)
    assert rc == 0
    assert env["command"] == "schema"
    assert env["status"] == "ok"
    r = env["result"]
    assert set(r) == {
        "tool",
        "commands",
        "error_codes",
        "warning_codes",
        "exit_codes",
        "supported_ops",
        "runtime",
    }
    assert r["tool"]["name"] == "coreai-onnx"
    assert r["tool"]["schema_version"] == 1
    assert r["tool"]["version"]  # non-empty


def test_schema_lists_all_commands_and_flags(capsys):
    _, env = _run_json(["schema", "--json"], capsys)
    cmds = {c["name"]: c for c in env["result"]["commands"]}
    assert set(cmds) == {"convert", "inspect", "verify", "schema"}
    convert_flags = {o["flag"] for o in cmds["convert"]["options"]}
    assert {
        "--output",
        "--no-optimize",
        "--name",
        "--no-validate",
        "--no-verify",
        "--repair",
        "--rtol",
        "--atol",
        "--seed",
    } <= convert_flags
    # the SUPPRESSed per-subcommand --json duplicate must NOT leak
    assert "--json" not in convert_flags
    assert [a["name"] for a in cmds["convert"]["arguments"]] == ["model"]
    assert [a["name"] for a in cmds["verify"]["arguments"]] == ["model", "aimodel"]
    # global --json is described once at tool level
    glob = {o["flag"] for o in env["result"]["tool"]["global_options"]}
    assert "--json" in glob


def test_schema_supported_ops(capsys):
    _, env = _run_json(["schema", "--json"], capsys)
    ops = env["result"]["supported_ops"]
    assert len(ops) > 100
    assert ops == sorted(ops)
    assert "Conv" in ops
    assert "Add" in ops
    assert "Det" not in ops


def test_schema_codes_match_tables(capsys):
    from coreai_onnx import _service

    _, env = _run_json(["schema", "--json"], capsys)
    r = env["result"]
    assert {e["code"] for e in r["error_codes"]} == set(_service._ERROR_CODES)
    assert {w["code"] for w in r["warning_codes"]} == set(_service._WARNING_CODES)
    assert {e["code"] for e in r["exit_codes"]} == set(_service._EXIT_CODES)


def test_schema_human_mode(capsys):
    rc = main(["schema"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "convert" in out
    assert "schema" in out
    assert "\x1b[" not in out  # still no ANSI when piped
