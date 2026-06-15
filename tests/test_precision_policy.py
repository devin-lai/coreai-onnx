# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""The default precision verdict policy in _service._run_precision_check.

Drives the orchestration with a stubbed ``coreai_onnx.verify`` so the policy -
fast default-unit primary, fp32 cpu_only fallback only on a miss, benign-noise
and hardware-divergence demotion - is tested without the native runtime. ORT is
bypassed by passing inputs/expected.
"""

import argparse

import pytest

import coreai_onnx
from coreai_onnx import OutputReport, VerifyReport, _service


def _args(**kw):
    base = {
        "rtol": None,
        "atol": None,
        "min_psnr": None,
        "seed": 0,
        "name": "main",
        "compute_unit": None,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def _report(passed, *, elementwise=None, psnr=120.0, name="y"):
    """A one-output VerifyReport. elementwise defaults to `passed`."""
    if elementwise is None:
        elementwise = passed
    out = OutputReport(
        name=name,
        max_abs_error=0.0 if elementwise else 1.0,
        max_rel_error=0.0 if elementwise else 1.0,
        psnr=psnr,
        passed=passed,
        elementwise_passed=elementwise,
    )
    return VerifyReport(outputs=[out], passed=passed)


def _stub_verify(by_unit, calls):
    """Build an async verify stub returning/raising per compute_unit."""

    async def _fake(model, aimodel, *, compute_unit=None, **_kw):
        calls.append(compute_unit)
        outcome = by_unit[compute_unit]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _fake


def _run(monkeypatch, by_unit, args):
    calls = []
    monkeypatch.setattr(coreai_onnx, "verify", _stub_verify(by_unit, calls))
    pc = _service._run_precision_check(
        object(), "m.aimodel", args, inputs={}, expected={}
    )
    return pc, calls


def test_primary_pass_no_fallback(monkeypatch):
    pc, calls = _run(monkeypatch, {None: _report(True)}, _args())
    assert pc.passed is True
    assert pc.effective_unit is None
    assert calls == [None]  # default unit only; no cpu_only fallback
    assert pc.warnings == []


def test_benign_noise_passes_with_warning(monkeypatch):
    # Primary passes, but only via the PSNR floor (elementwise failed).
    pc, calls = _run(
        monkeypatch, {None: _report(True, elementwise=False, psnr=112.0)}, _args()
    )
    assert pc.passed is True
    assert pc.effective_unit is None
    assert calls == [None]
    assert [w["code"] for w in pc.warnings] == ["precision_benign_noise"]


def test_hardware_divergence_fallback_to_cpu_only(monkeypatch):
    # Default unit fails (float16); fp32 cpu_only is faithful -> pass + warning.
    pc, calls = _run(
        monkeypatch,
        {None: _report(False), "cpu_only": _report(True)},
        _args(),
    )
    assert pc.passed is True
    assert pc.effective_unit == "cpu_only"
    assert calls == [None, "cpu_only"]
    assert [w["code"] for w in pc.warnings] == ["precision_hardware_divergence"]


def test_native_abort_on_primary_falls_back(monkeypatch):
    # Primary aborts (raises); fp32 confirms fidelity -> pass + warning.
    pc, calls = _run(
        monkeypatch,
        {None: RuntimeError("runtime aborted"), "cpu_only": _report(True)},
        _args(),
    )
    assert pc.passed is True
    assert pc.effective_unit == "cpu_only"
    assert calls == [None, "cpu_only"]
    assert [w["code"] for w in pc.warnings] == ["precision_hardware_divergence"]


def test_real_failure_fails_on_fp32_too(monkeypatch):
    pc, calls = _run(
        monkeypatch,
        {None: _report(False), "cpu_only": _report(False)},
        _args(),
    )
    assert pc.passed is False
    assert pc.effective_unit == "cpu_only"
    assert calls == [None, "cpu_only"]
    # A genuine failure is not dressed up as a hardware-divergence pass.
    assert all(w["code"] != "precision_hardware_divergence" for w in pc.warnings)


def test_cpu_only_abort_propagates(monkeypatch):
    # Both the default unit and the fp32 fallback abort -> raise (caller maps it
    # to precision_check_error). The fast path failing must not hide a fp32 crash.
    by_unit = {None: _report(False), "cpu_only": RuntimeError("fp32 aborted too")}
    calls = []
    monkeypatch.setattr(coreai_onnx, "verify", _stub_verify(by_unit, calls))
    with pytest.raises(RuntimeError, match="fp32 aborted too"):
        _service._run_precision_check(object(), "m.aimodel", _args(), inputs={}, expected={})


def test_explicit_unit_is_verdict_without_fallback(monkeypatch):
    # An explicit --compute-unit is honored exactly: a failure there is the
    # verdict, with no fp32 fallback (the user asked about that unit).
    pc, calls = _run(
        monkeypatch,
        {"gpu": _report(False), "cpu_only": _report(True)},
        _args(compute_unit="gpu"),
    )
    assert pc.passed is False
    assert pc.effective_unit == "gpu"
    assert calls == ["gpu"]  # cpu_only never consulted


def test_explicit_min_psnr_overrides_default_floor(monkeypatch):
    # User-supplied --min-psnr flows through as the floor used by verify().
    seen = {}

    async def _fake(model, aimodel, *, compute_unit=None, min_psnr=None, **_kw):
        seen["min_psnr"] = min_psnr
        return _report(True)

    monkeypatch.setattr(coreai_onnx, "verify", _fake)
    _service._run_precision_check(
        object(), "m.aimodel", _args(min_psnr=70.0), inputs={}, expected={}
    )
    assert seen["min_psnr"] == 70.0


def test_default_floor_applied_when_min_psnr_unset(monkeypatch):
    seen = {}

    async def _fake(model, aimodel, *, compute_unit=None, min_psnr=None, **_kw):
        seen["min_psnr"] = min_psnr
        return _report(True)

    monkeypatch.setattr(coreai_onnx, "verify", _fake)
    _service._run_precision_check(object(), "m.aimodel", _args(), inputs={}, expected={})
    assert seen["min_psnr"] == _service._DEFAULT_BENIGN_PSNR_DB
