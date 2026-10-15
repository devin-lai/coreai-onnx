# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for shared command helpers."""

from pathlib import Path

import pytest

import coreai_onnx
from coreai_onnx import _service
from coreai_onnx._cli import _build_parser

from .helpers import relu_model_file

_LICENSE = "BSD-3-Clause"
_REPOSITORY = "https://github.com/devin-lai/coreai-onnx"


def test_asset_metadata_identifies_converter():
    metadata = _service._asset_metadata()

    assert metadata.author == "coreai-onnx contributors"
    assert metadata.license == _LICENSE
    assert metadata.model_description == f"Converted with coreai-onnx: {_REPOSITORY}"
    assert metadata.creator_defined_metadata["coreai_onnx.repository"] == _REPOSITORY
    assert metadata.creator_defined_metadata["coreai_onnx.license"] == _LICENSE


@pytest.mark.parametrize("version", ["26.1", "14.7", "", "invalid"])
def test_verify_rejects_macos_without_runtime(monkeypatch, version):
    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_service.platform, "mac_ver", lambda: (version, (), ""))
    args = _build_parser().parse_args(["verify", "missing.onnx", "missing.aimodel"])
    outcome, _ = _service._execute_command("verify", args)
    assert outcome.exit_code == 2
    assert outcome.error["code"] == "platform_unsupported"


def test_verify_invalid_model_has_invalid_model_code(tmp_path, monkeypatch):
    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_service.platform, "mac_ver", lambda: ("27.0", (), ""))
    model = tmp_path / "invalid.onnx"
    model.write_bytes(b"not a protobuf")
    args = _build_parser().parse_args(["verify", str(model), "missing.aimodel"])
    outcome, _ = _service._execute_command("verify", args)
    assert outcome.error["code"] == "invalid_model_file"


@pytest.fixture
def staged_conversion(tmp_path, monkeypatch):
    class Program:
        def save_asset(self, path, **kwargs):
            path.mkdir()
            (path / "metadata.json").write_text("{}")
            (path / "program.mlirb").write_bytes(b"new model")

    program = Program()
    monkeypatch.setattr(coreai_onnx, "convert", lambda *a, **kw: program)
    monkeypatch.setattr(_service, "optimize_program", lambda p: None)
    monkeypatch.setattr(_service, "_asset_metadata", lambda: None)
    output = tmp_path / "model.aimodel"
    output.mkdir()
    (output / "metadata.json").write_text("{}")
    (output / "program.mlirb").write_bytes(b"original model")
    args = _build_parser().parse_args(
        [
            "convert",
            relu_model_file(tmp_path),
            "-o",
            str(output),
            "--no-validate",
            "--no-verify",
        ]
    )
    return program, output, args


@pytest.mark.parametrize("step", ["optimize", "save"])
def test_failed_conversion_preserves_existing_asset(
    staged_conversion, monkeypatch, step
):
    program, output, args = staged_conversion

    def fail(*a, **kw):
        if step == "save":
            a[0].mkdir()
            (a[0] / "partial").write_bytes(b"partial")
        raise RuntimeError("compiler failed")

    if step == "optimize":
        monkeypatch.setattr(_service, "optimize_program", fail)
    else:
        monkeypatch.setattr(program, "save_asset", fail)
    outcome, _ = _service._execute_command("convert", args)
    assert outcome.error["code"] == "compiler_failed"
    assert (output / "program.mlirb").read_bytes() == b"original model"
    assert sorted(p.name for p in output.parent.iterdir()) == [
        "model.aimodel",
        "relu.onnx",
    ]


@pytest.mark.parametrize("name", ["model.aimodel", "previous"])
def test_saved_conversion_replaces_asset(staged_conversion, name):
    _, output, args = staged_conversion
    output = output.rename(output.with_name(name)) if name != output.name else output
    args.output = str(output)
    outcome, _ = _service._execute_command("convert", args)
    assert outcome.error is None
    assert (output / "program.mlirb").read_bytes() == b"new model"
    assert sorted(p.name for p in output.parent.iterdir()) == [
        name,
        "relu.onnx",
    ]


def test_failed_asset_install_restores_previous_asset(staged_conversion, monkeypatch):
    _, output, args = staged_conversion
    rename = Path.rename

    def fail_install(self, target):
        if self.name == output.name and self != output and target == output:
            raise OSError("cannot install new asset")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_install)
    outcome, _ = _service._execute_command("convert", args)
    assert outcome.error["code"] == "io_error"
    assert (output / "program.mlirb").read_bytes() == b"original model"


def test_failed_asset_rollback_keeps_backup(staged_conversion, monkeypatch):
    _, output, args = staged_conversion
    rename = Path.rename

    def fail_restore(self, target):
        if target == output:
            raise OSError("destination became unwritable")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_restore)
    outcome, _ = _service._execute_command("convert", args)
    assert outcome.error["code"] == "io_error"
    backups = list(output.parent.glob(".coreai_onnx_*/previous/program.mlirb"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"original model"
    assert str(backups[0].parent) in outcome.error["message"]


def test_convert_skips_precision_on_old_macos(staged_conversion, monkeypatch):
    _, _, args = staged_conversion
    args.no_verify = False
    monkeypatch.setattr(_service.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_service.platform, "mac_ver", lambda: ("26.0", (), ""))
    monkeypatch.setattr(_service, "_onnxruntime_available", lambda: True)

    def unexpected_check(*a, **kw):
        pytest.fail("precision check must not run without the native runtime")

    monkeypatch.setattr(_service, "_run_precision_check", unexpected_check)
    outcome, _ = _service._execute_command("convert", args)
    assert outcome.error is None
    assert [w["code"] for w in outcome.warnings] == ["platform_no_runtime"]
