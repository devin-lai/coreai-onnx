# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Crash isolation of the native .aimodel execution step (_verify_worker).

The native runtime can abort the process (not raise) on a program it cannot
execute on the selected compute unit. These tests pin the contract that such an
abort - and any other abnormal child exit - is turned into a CoreaiOnnxError the
service layer maps to precision_check_error, without needing to provoke a real
abort. One runtime-gated test proves isolation is transparent for a good model.
"""

import json
import subprocess

import numpy as np
import pytest

from coreai_onnx import CoreaiOnnxError
from coreai_onnx import _verify_worker as vw


def test_crash_message_signal_names_signal_and_recovery():
    msg = vw._crash_message(-6, "MPSGraphTensorData.mm:680: failed assertion ...")
    assert "aborted" in msg
    assert "SIGABRT" in msg
    # points the caller at the fp32 path rather than a blind retry
    assert "cpu_only" in msg
    # the native diagnostic (last stderr line) is surfaced
    assert "failed assertion" in msg


def test_crash_message_nonzero_exit_is_abnormal():
    msg = vw._crash_message(1, "")
    assert "abnormally" in msg
    assert "exit code 1" in msg


def test_crash_message_unknown_signal_does_not_raise():
    # -99 is not a real signal number; must degrade gracefully, not crash.
    msg = vw._crash_message(-99, "")
    assert "aborted" in msg


def test_interpret_reads_outputs(tmp_path):
    np.savez(tmp_path / "outputs.npz", np.arange(6, dtype=np.float32).reshape(2, 3))
    got = vw._interpret(0, "", tmp_path, ["y"])
    assert list(got) == ["y"]
    np.testing.assert_array_equal(got["y"], np.arange(6, dtype=np.float32).reshape(2, 3))


def test_interpret_surfaces_child_handled_error(tmp_path):
    (tmp_path / "error.json").write_text(
        json.dumps({"type": "ValueError", "message": "bad asset"})
    )
    with pytest.raises(CoreaiOnnxError, match="ValueError: bad asset"):
        vw._interpret(1, "", tmp_path, ["y"])


def test_interpret_signal_death_becomes_structured_error(tmp_path):
    # No outputs, no error.json, killed by signal -> a crash, mapped to error.
    with pytest.raises(CoreaiOnnxError, match="aborted"):
        vw._interpret(-6, "boom", tmp_path, ["y"])


def test_interpret_zero_exit_without_outputs_is_a_crash(tmp_path):
    # Defensive: a 0 exit that produced no outputs is still a protocol failure.
    with pytest.raises(CoreaiOnnxError):
        vw._interpret(0, "", tmp_path, ["y"])


def test_run_asset_isolated_maps_signal_death(tmp_path, monkeypatch):
    """End-to-end through run_asset_isolated, with the child faked as killed."""

    def fake_run(_cmd, **_kw):
        return subprocess.CompletedProcess(_cmd, returncode=-6, stdout="", stderr="x")

    monkeypatch.setattr(vw.subprocess, "run", fake_run)
    with pytest.raises(CoreaiOnnxError, match="aborted"):
        vw.run_asset_isolated(
            tmp_path / "m.aimodel",
            {"x": np.zeros((2, 3), dtype=np.float32)},
            ["y"],
            "main",
            None,
        )


def test_run_asset_isolated_real_subprocess_reports_bad_asset(tmp_path):
    """The real worker subprocess runs and reports a structured error for a
    missing .aimodel (exercises the spawn/IPC path without a real abort)."""
    with pytest.raises(CoreaiOnnxError):
        vw.run_asset_isolated(
            tmp_path / "does_not_exist.aimodel",
            {"x": np.zeros((2, 3), dtype=np.float32)},
            ["y"],
            "main",
            None,
        )
