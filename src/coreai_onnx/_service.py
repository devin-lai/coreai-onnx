# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared command core for the CLI and the MCP server.

Runners compute an _Outcome (never print); _envelope() serializes it into
the schema_version-1 JSON envelope; the code tables here are the frozen,
append-only contract documented in docs/cli.md. coreai_onnx._cli renders
outcomes for humans and owns the argparse parser; coreai_onnx._mcp wraps
the same runners as MCP tools. Both surfaces emit identical envelopes
because both go through _execute_command()/_envelope() in this module.

One deliberate inversion: _run_schema introspects the live argparse parser
(the single source of truth for the command surface) via a lazy import of
coreai_onnx._cli._build_parser, mirroring this codebase's existing pattern
of function-level imports for optional/cyclic edges.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import platform
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import onnx
from google.protobuf.message import DecodeError
from onnx.checker import ValidationError

from . import _coverage
from .errors import (
    ConversionError,
    CoreaiOnnxError,
    ModelValidationError,
    UnsupportedOpError,
)

_ORT_MISSING_MSG = (
    "onnxruntime is not installed; skipping ONNX Runtime validation and the "
    'precision check. Install "coreai-onnx[verify]" to enable them.'
)

_PRECISION_FAILED_HINT = (
    "Inspect result.precision: a high PSNR (e.g. > 60 dB) with elementwise "
    "failures usually means benign accumulation noise on large-magnitude "
    "outputs — consider --min-psnr. GPU/ANE execute in float16; "
    "`--compute-unit cpu_only` checks the fp32 path that proves conversion "
    "fidelity. See docs on benign causes."
)

_PRECISION_ERROR_HINT = (
    "The .aimodel was written and the conversion is unaffected; only the "
    "post-conversion check could not complete. If the native runtime aborted "
    "executing the program on the selected compute unit, re-check with "
    "`--compute-unit cpu_only` (the fp32 path) rather than retrying as-is."
)

# Default PSNR floor (dB) for accepting an output that fails the elementwise
# tolerance as benign accumulation noise, when the caller did not pass an
# explicit --min-psnr. 50 dB is ~0.3% RMS error: the docs anchor 60 dB as a
# faithful conversion and 100+ as near-perfect, and across every real-world
# model we have measured, faithful conversions sit at >=57 dB while genuine
# lowering bugs land below ~40 dB (and a localized large error drags PSNR down
# with them), so 50 dB separates the two with margin. Acceptance still requires
# the non-finite pattern to match the reference (a NaN/Inf divergence is never
# benign). Tighten or loosen with --min-psnr / --rtol / --atol.
_DEFAULT_BENIGN_PSNR_DB = 50.0

# ---------------------------------------------------------------------------
# contract tables + envelope (the frozen schema_version-1 surface)
# ---------------------------------------------------------------------------

# Single source of truth for the stable code contracts (all three tables:
# error, warning, and exit codes). The schema command, docs drift tests, and
# the _error/_warning constructors all read these. Codes are append-only;
# meanings are stable *descriptions* (exposed by the schema dump) and may be
# clarified — the per-instance "message" in an envelope is free-form context
# and intentionally not tied to these strings. See docs/cli.md.
_ERROR_CODES: dict[str, dict[str, str | None]] = {
    "unsupported_ops": {
        "meaning": "The model contains ops with no Core AI lowering.",
        "details": "missing: dict mapping op key to example node names",
    },
    "model_validation_failed": {
        "meaning": "The input model failed to load or run on ONNX Runtime.",
        "details": None,
    },
    "conversion_failed": {
        "meaning": "A lowering failed while converting the model.",
        "details": "node_name and op_key, when known",
    },
    "compiler_failed": {
        "meaning": "The Core AI compiler failed to optimize or save the program.",
        "details": "exception_type",
    },
    "precision_check_failed": {
        "meaning": "Outputs exceeded tolerance vs ONNX Runtime"
        " (for convert, the .aimodel was still written).",
        "details": None,
    },
    "precision_check_error": {
        "meaning": "The precision check could not run (e.g. the Core AI runtime "
        "aborted executing the .aimodel on the selected compute unit); for "
        "convert the .aimodel was still written.",
        "details": "exception_type",
    },
    "invalid_model_file": {
        "meaning": "The file is not a valid ONNX model.",
        "details": None,
    },
    "io_error": {
        "meaning": "A file could not be read or written.",
        "details": None,
    },
    "platform_unsupported": {
        "meaning": "verify requires macOS 27+ with the Core AI runtime.",
        "details": None,
    },
}

_WARNING_CODES: dict[str, str] = {
    "onnxruntime_missing": "onnxruntime is not installed; ONNX Runtime "
    "validation and the precision check were skipped.",
    "platform_no_runtime": "The precision check requires macOS 27+ with the "
    "Core AI runtime and was skipped; the .aimodel was still written.",
    "reference_nonfinite": "The ONNX Runtime reference output contains "
    "non-finite values (the input model itself produces NaN/Inf on the random "
    "probe input); parity at those positions is checked by NaN/Inf mask and "
    "the error metrics cover only the finite region.",
    "precision_benign_noise": "An output failed the elementwise tolerance but "
    "was accepted as benign accumulation noise: its PSNR is at or above the "
    "benign floor (default 50 dB; override with --min-psnr) and its non-finite "
    "pattern matches the reference. The conversion is faithful; tighten with "
    "--rtol/--atol or --min-psnr to treat such outputs as failures.",
    "precision_hardware_divergence": "The .aimodel diverged from ONNX Runtime "
    "(or could not execute) on the runtime's default compute unit, but is "
    "faithful on the fp32 CPU path - the divergence is float16 GPU/ANE hardware "
    "behavior, not a conversion error. The verdict reflects the fp32 path; pin "
    "--compute-unit gpu/ane to inspect the hardware path, or ship the model "
    "with cpu_only specialization if it needs fp32.",
}

_EXIT_CODES: dict[int, str] = {
    0: "Success.",
    1: "Failure: bad model, unsupported ops, conversion or verification failure.",
    2: "Usage error (argparse, on stderr) or platform error.",
    3: "The .aimodel was written but the precision check failed or could not run.",
}


@dataclass
class _Outcome:
    """Result of running one CLI command: everything needed to render it.

    ``result`` is a JSON-ready dict (the command's payload), ``warnings`` a
    list of ``{"code", "message"}`` dicts, ``error`` a
    ``{"code", "message", "details", "hint"}`` dict or None. Runners never
    print; renderers never compute.
    """

    exit_code: int
    result: dict | None = None
    warnings: list[dict] = field(default_factory=list)
    error: dict | None = None


SCHEMA_VERSION = 1


def _envelope(command: str, outcome: _Outcome) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": "error" if outcome.error is not None else "ok",
        "result": outcome.result,
        "warnings": outcome.warnings,
        "error": outcome.error,
    }


def _error(
    code: str, message: str, details: dict | None = None, hint: str | None = None
) -> dict:
    if code not in _ERROR_CODES:
        raise RuntimeError(f"undeclared error code: {code}")
    return {"code": code, "message": message, "details": details, "hint": hint}


def _warning(code: str, message: str) -> dict:
    if code not in _WARNING_CODES:
        raise RuntimeError(f"undeclared warning code: {code}")
    return {"code": code, "message": message}


def _error_from_exception(exc: Exception) -> dict:
    """Map an exception to a stable structured error. Codes are append-only;
    see docs/cli.md for the contract."""
    if isinstance(exc, UnsupportedOpError):
        return _error(
            "unsupported_ops",
            str(exc),
            details={"missing": exc.missing},
            hint=(
                "Register a custom lowering with "
                "@converter.register_onnx_lowering, or run "
                "`coreai-onnx inspect <model>` for a full coverage report."
            ),
        )
    if isinstance(exc, ConversionError):
        return _error(
            "conversion_failed",
            str(exc),
            details={"node_name": exc.node_name, "op_key": exc.op_key},
        )
    if isinstance(exc, ModelValidationError):
        return _error("model_validation_failed", str(exc))
    if isinstance(exc, ValidationError | DecodeError):
        return _error("invalid_model_file", str(exc))
    if isinstance(exc, OSError):
        return _error("io_error", str(exc))
    # Remaining CoreaiOnnxError subclasses without a more specific shape.
    return _error("conversion_failed", str(exc))


def _onnxruntime_available() -> bool:
    """True if onnxruntime can be imported (it is an optional [verify] extra)."""
    import importlib.util

    return importlib.util.find_spec("onnxruntime") is not None


# ---------------------------------------------------------------------------
# runners (compute, no printing)
# ---------------------------------------------------------------------------


def _json_float(v: float) -> float | str:
    """JSON has no literals for non-finite floats (json.dumps would emit bare
    Infinity/-Infinity/NaN tokens); serialize those as the strings "inf",
    "-inf", "nan" so the envelope stays strictly parseable."""
    if math.isfinite(v):
        return v
    if math.isnan(v):
        return "nan"
    return "inf" if v > 0 else "-inf"


def _verify_result_dict(
    report, args: argparse.Namespace, *, compute_unit: str | None
) -> dict:
    """Serialize a VerifyReport. rtol/atol/min_psnr echo what was requested on
    the command line (null means the default chosen inside verify()).
    ``compute_unit`` is the *effective* unit that produced this verdict - the
    explicit --compute-unit, or "cpu_only" when the default check fell back to
    the fp32 path, or null when the runtime's default unit was authoritative.
    Non-finite metrics (psnr of +/-inf, NaN/Inf error maxima) serialize as
    strings."""
    return {
        "passed": report.passed,
        "rtol": _json_float(args.rtol) if args.rtol is not None else None,
        "atol": _json_float(args.atol) if args.atol is not None else None,
        "min_psnr": _json_float(args.min_psnr) if args.min_psnr is not None else None,
        "compute_unit": compute_unit,
        "seed": args.seed,
        "outputs": [
            {
                "name": o.name,
                "max_abs_error": _json_float(o.max_abs_error),
                "max_rel_error": _json_float(o.max_rel_error),
                "psnr": _json_float(o.psnr),
                "passed": o.passed,
                "expected_nonfinite": o.expected_nonfinite,
            }
            for o in report.outputs
        ],
    }


def _reference_nonfinite_warning(report) -> dict | None:
    """A reference_nonfinite warning naming the affected outputs, or None."""
    affected = [o.name for o in report.outputs if o.expected_nonfinite]
    if not affected:
        return None
    return _warning(
        "reference_nonfinite",
        "the ONNX Runtime reference contains non-finite values for "
        f"output(s) {', '.join(affected)} — the input model itself produces "
        "NaN/Inf on the random probe input; parity at those positions is "
        "checked by NaN/Inf mask",
    )


@dataclass
class _PrecisionCheck:
    """The authoritative precision verdict plus the unit/warnings that explain it."""

    report: object  # VerifyReport
    warnings: list[dict]
    effective_unit: str | None
    passed: bool


def _benign_noise_warning(report) -> dict | None:
    """Warn when an output passed only via the PSNR floor, not elementwise."""
    benign = [o.name for o in report.outputs if o.passed and not o.elementwise_passed]
    if not benign:
        return None
    worst = min(
        o.psnr
        for o in report.outputs
        if o.passed and not o.elementwise_passed and math.isfinite(o.psnr)
    )
    return _warning(
        "precision_benign_noise",
        f"output(s) {', '.join(benign)} failed the elementwise tolerance but "
        f"were accepted as benign accumulation noise (PSNR ≥ {_DEFAULT_BENIGN_PSNR_DB:g} "
        f"dB floor; worst {worst:.1f} dB) — the conversion is faithful",
    )


def _hardware_divergence_warning(primary, primary_error: Exception | None) -> dict:
    """Warn that the default unit diverged but the fp32 verdict is faithful."""
    if primary_error is not None:
        detail = (
            "could not execute on the runtime's default compute unit "
            f"({type(primary_error).__name__})"
        )
    else:
        failed = ", ".join(o.name for o in primary.outputs if not o.passed)
        detail = f"diverged from ONNX Runtime on the runtime's default compute unit ({failed})"
    return _warning(
        "precision_hardware_divergence",
        f"the .aimodel {detail}, but is faithful on the fp32 CPU path; the "
        "divergence is float16 GPU/ANE hardware behavior, not a conversion "
        "error. The verdict below reflects the fp32 (cpu_only) path.",
    )


def _run_precision_check(
    model, aimodel_path, args: argparse.Namespace, *, inputs=None, expected=None
) -> _PrecisionCheck:
    """Run the post-conversion precision check with the default-verdict policy.

    The verdict measures *conversion fidelity*. With no --compute-unit, the
    check runs first on the runtime's default unit (fast); on success that is
    the verdict — the common path, with zero added cost. Only if it fails or the
    native runtime aborts does it fall back to the deterministic fp32 cpu_only
    path (slower, so it runs only for the few models that need adjudication). A
    model faithful on fp32 but divergent on the default float16 GPU/ANE unit
    passes with a precision_hardware_divergence warning; benign large-magnitude
    accumulation noise passes via the PSNR floor with a precision_benign_noise
    warning. An explicit --compute-unit is honored exactly, with no fallback.

    ONNX Runtime runs once: inputs/expected are generated here if not supplied
    and reused across the primary and fallback runs. Raises (the caller maps it
    to precision_check_error) when the authoritative run cannot complete.
    """
    from . import generate_inputs
    from . import verify as _verify
    from ._verify import _run_onnxruntime

    if inputs is None:
        inputs = generate_inputs(model, seed=args.seed)
    if expected is None:
        expected = _run_onnxruntime(model, inputs)
    floor = args.min_psnr if args.min_psnr is not None else _DEFAULT_BENIGN_PSNR_DB

    def _run(unit: str | None):
        return asyncio.run(
            _verify(
                model,
                aimodel_path,
                rtol=args.rtol,
                atol=args.atol,
                min_psnr=floor,
                seed=args.seed,
                inputs=inputs,
                expected=expected,
                entrypoint=args.name,
                compute_unit=unit,
                isolate_execution=True,
            )
        )

    def _present(*candidates: dict | None) -> list[dict]:
        return [w for w in candidates if w is not None]

    # An explicit unit is the verdict, as asked — no fallback.
    if args.compute_unit is not None:
        report = _run(args.compute_unit)
        return _PrecisionCheck(
            report, _present(_benign_noise_warning(report)), args.compute_unit, report.passed
        )

    # Default: fast path on the runtime's choice; fall back to fp32 only on miss.
    primary_error: Exception | None = None
    try:
        primary = _run(None)
        if primary.passed:
            return _PrecisionCheck(
                primary, _present(_benign_noise_warning(primary)), None, True
            )
    except Exception as exc:
        primary = None
        primary_error = exc

    fp32 = _run("cpu_only")  # may raise -> precision_check_error
    if not fp32.passed:
        return _PrecisionCheck(
            fp32, _present(_benign_noise_warning(fp32)), "cpu_only", False
        )
    warnings = _present(
        _hardware_divergence_warning(primary, primary_error),
        _benign_noise_warning(fp32),
    )
    return _PrecisionCheck(fp32, warnings, "cpu_only", True)


def _run_inspect(args: argparse.Namespace) -> _Outcome:
    model = onnx.load(args.model)
    report = _coverage.analyze(model)
    supported = _coverage.supported_ops()
    result = {
        "model_path": args.model,
        "total_nodes": report.total_nodes,
        "convertible": report.convertible,
        "ops": [
            {"op": op, "count": count, "supported": op in supported}
            for op, count in sorted(report.op_histogram.items())
        ],
        "unsupported": sorted(report.unsupported),
    }
    return _Outcome(exit_code=0 if report.convertible else 1, result=result)


def _replace_existing_aimodel(out_path: Path) -> None:
    """Clear a previous .aimodel bundle so re-converting to the same output
    path works (save_asset itself cannot overwrite). Only something that is
    recognizably an .aimodel bundle (a directory with metadata.json and a
    .mlirb program) is removed; anything else raises OSError -> io_error,
    so a mistyped -o can never delete unrelated files."""
    if not out_path.exists():
        return
    is_bundle = (
        out_path.is_dir()
        and (out_path / "metadata.json").is_file()
        and any(out_path.glob("*.mlirb"))
    )
    if not is_bundle:
        raise OSError(
            f"output path '{out_path}' exists and is not an .aimodel bundle; "
            "refusing to replace it — choose another path or remove it first"
        )
    shutil.rmtree(out_path)


def _run_convert(args: argparse.Namespace) -> _Outcome:
    from . import convert as _convert

    # Load once: the proto feeds validation, the summary, conversion, and the
    # precision check.
    model = onnx.load(args.model)

    # Optional auto-repair: rewrite known Core AI runtime limitations on the
    # model up front (e.g. promote float16 to float32) so validation, the
    # summary, conversion, and the precision check all run on the repaired
    # graph. Repairs are semantics-preserving; the precision check below is the
    # safety net that rejects any rewrite that would change results.
    repairs: list[dict] = []
    if getattr(args, "repair", False):
        from . import _repair

        model, records = _repair.apply_repairs(model)
        repairs = [record.as_dict() for record in records]

    report = _coverage.analyze(model)

    have_ort = _onnxruntime_available()
    do_validate = not args.no_validate
    do_verify = not args.no_verify

    warnings: list[dict] = []
    # An onnxruntime-dependent step was requested but the package is absent:
    # note it once and carry on - conversion itself never needs onnxruntime.
    if (do_validate or do_verify) and not have_ort:
        warnings.append(_warning("onnxruntime_missing", _ORT_MISSING_MSG))

    # Step 1: pre-conversion ONNX Runtime validation gate. A failure here is a
    # problem with the input model: validate_onnxruntime raises
    # ModelValidationError, which reaches main(), renders cleanly, and exits 1
    # - nothing is written. The reference outputs are reused below so ONNX
    # Runtime runs only once.
    inputs = None
    expected = None
    if do_validate and have_ort:
        from . import generate_inputs, validate_onnxruntime

        inputs = generate_inputs(model, seed=args.seed)
        expected = validate_onnxruntime(model, inputs=inputs)

    # Step 2: convert.
    program = _convert(model, entrypoint_name=args.name)

    # Step 3: optimize + save. Native Core AI compiler failures are not
    # CoreaiOnnxError; capture them as a structured error (their message
    # carries the MLIR diagnostic) instead of letting a binding traceback
    # reach the terminal.
    out_path = Path(args.output)
    _replace_existing_aimodel(out_path)
    try:
        if not args.no_optimize:
            program.optimize()
        program.save_asset(out_path)
    except Exception as exc:
        return _Outcome(
            exit_code=1,
            warnings=warnings,
            error=_error(
                "compiler_failed",
                f"Core AI compiler failed to optimize/save the program: "
                f"{type(exc).__name__}: {exc}",
                details={"exception_type": type(exc).__name__},
            ),
        )

    result = {
        "output_path": str(out_path),
        "total_nodes": report.total_nodes,
        "optimized": not args.no_optimize,
        "validated": bool(do_validate and have_ort),
        "repairs": repairs,
        "precision": None,
    }

    # Step 4: precision check (auto). Needs onnxruntime + the macOS Core AI
    # runtime; skip gracefully off-platform - the .aimodel is already written.
    if not (do_verify and have_ort):
        return _Outcome(exit_code=0, result=result, warnings=warnings)
    if platform.system() != "Darwin":
        warnings.append(
            _warning(
                "platform_no_runtime",
                "the precision check requires macOS 27+ with the Core AI "
                "runtime; skipping (the .aimodel was still written).",
            )
        )
        return _Outcome(exit_code=0, result=result, warnings=warnings)

    try:
        pc = _run_precision_check(
            model, out_path, args, inputs=inputs, expected=expected
        )
    except Exception as exc:
        # The asset is on disk and conversion succeeded; the precision check
        # itself could not complete - including when the native runtime aborts
        # executing the program (isolate_execution runs it in a child process so
        # that abort returns here instead of killing us). Surface it distinctly
        # (exit 3), not as a conversion failure (exit 1).
        return _Outcome(
            exit_code=3,
            result=result,
            warnings=warnings,
            error=_error(
                "precision_check_error",
                str(exc),
                details={"exception_type": type(exc).__name__},
                hint=_PRECISION_ERROR_HINT,
            ),
        )

    nonfinite = _reference_nonfinite_warning(pc.report)
    if nonfinite is not None:
        warnings.append(nonfinite)
    warnings.extend(pc.warnings)
    result["precision"] = _verify_result_dict(pc.report, args, compute_unit=pc.effective_unit)
    if pc.passed:
        return _Outcome(exit_code=0, result=result, warnings=warnings)
    return _Outcome(
        exit_code=3,
        result=result,
        warnings=warnings,
        error=_error(
            "precision_check_failed",
            "outputs exceeded tolerance; the .aimodel was still written",
            hint=_PRECISION_FAILED_HINT,
        ),
    )


def _parser_actions(parser: argparse.ArgumentParser) -> tuple[argparse.Action, ...]:
    """Return a parser's actions behind one typed private-argparse boundary."""
    return tuple(getattr(parser, "_actions", ()))


def _subparser_choices(
    action: argparse.Action,
) -> Mapping[str, argparse.ArgumentParser] | None:
    choices = getattr(action, "choices", None)
    if not isinstance(choices, Mapping):
        return None
    if not all(
        isinstance(name, str) and isinstance(value, argparse.ArgumentParser)
        for name, value in choices.items()
    ):
        return None
    return cast("Mapping[str, argparse.ArgumentParser]", choices)


def _is_boolean_flag(action: argparse.Action) -> bool:
    return (
        action.nargs == 0
        and isinstance(action.const, bool)
        and isinstance(action.default, bool)
    )


def _describe_actions(parser: argparse.ArgumentParser) -> tuple[list, list]:
    """Describe a parser's positionals and options for the schema dump.

    Reads argparse's private _actions: there is no public introspection API,
    and these attributes have been stable since 2.7. SUPPRESSed help entries
    (the per-subcommand --json duplicates) are skipped.
    """
    arguments: list[dict] = []
    options: list[dict] = []
    for action in _parser_actions(parser):
        if action.dest == "help":
            continue
        if action.help == argparse.SUPPRESS:
            continue
        if _subparser_choices(action) is not None:
            continue
        if not action.option_strings:
            arguments.append({"name": action.dest, "help": action.help or ""})
            continue
        if _is_boolean_flag(action):
            # Both boolean action kinds dump as "flag"; the real default
            # (False for store_true, True for store_false) is preserved.
            opt_type = "flag"
            default = action.default
        else:
            action_type = action.type
            opt_type = getattr(action_type, "__name__", "str")
            default = action.default
        options.append(
            {
                # Canonical form = the longest spelling ("-o"/"--output" ->
                # "--output"); every option here has a unique long form.
                "flag": max(action.option_strings, key=len),
                "type": opt_type,
                "default": default,
                "required": bool(action.required),
                "help": action.help or "",
            }
        )
    return arguments, options


def _run_schema(args: argparse.Namespace) -> _Outcome:
    from . import supported_ops
    from .__version__ import __version__

    # main() only hands runners the parsed Namespace; the parser object is
    # gone by now, so rebuild it — _build_parser() is cheap and side-effect
    # free, and walking the live parser keeps the dump drift-proof.
    # Imported lazily: _cli imports this module at top level.
    from ._cli import _build_parser

    parser = _build_parser()
    subparsers = next(
        choices
        for action in _parser_actions(parser)
        if (choices := _subparser_choices(action)) is not None
    )
    commands = []
    for name, sub in subparsers.items():
        arguments, options = _describe_actions(sub)
        commands.append(
            {
                "name": name,
                "summary": sub.description or "",
                "arguments": arguments,
                "options": options,
            }
        )

    _, global_options = _describe_actions(parser)

    result = {
        "tool": {
            "name": "coreai-onnx",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            # Kept in sync with pyproject.toml's description by a drift test.
            "description": "Convert ONNX models to Apple Core AI (.aimodel) "
            "— the AI-first successor to Core ML on iOS 27. Validated, "
            "precision-checked, agent-friendly, with automatic float16 repair "
            "and an MCP server.",
            "global_options": global_options,
        },
        "commands": commands,
        "error_codes": [
            {"code": code, "meaning": entry["meaning"], "details": entry["details"]}
            for code, entry in _ERROR_CODES.items()
        ],
        "warning_codes": [
            {"code": code, "meaning": meaning}
            for code, meaning in _WARNING_CODES.items()
        ],
        "exit_codes": [
            {"code": code, "meaning": meaning} for code, meaning in _EXIT_CODES.items()
        ],
        "supported_ops": sorted(supported_ops()),
        "runtime": {
            "convert_platforms": "any",
            "verify_platforms": "macOS 27+ with the Core AI runtime",
            "onnxruntime_required_for": ["validate", "verify"],
        },
    }
    return _Outcome(exit_code=0, result=result)


def _run_verify(args: argparse.Namespace) -> _Outcome:
    # Probing a coreai import here is useless: the CLI itself already imported
    # coreai at startup, so gate on the actual platform requirement instead.
    if platform.system() != "Darwin":
        return _Outcome(
            exit_code=2,
            error=_error(
                "platform_unsupported",
                "verify requires macOS 27+ with the Core AI runtime",
            ),
        )

    try:
        # onnx.load here (not inside the helper) so a missing/unreadable .onnx
        # raises OSError that propagates to main() as io_error, rather than
        # being reclassified as a precision_check_error.
        model = onnx.load(args.model)
        pc = _run_precision_check(model, args.aimodel, args)
    except OSError:
        raise
    except Exception as exc:
        return _Outcome(
            exit_code=1,
            error=_error(
                "precision_check_error",
                str(exc),
                details={"exception_type": type(exc).__name__},
                hint=_PRECISION_ERROR_HINT,
            ),
        )

    warnings = []
    nonfinite = _reference_nonfinite_warning(pc.report)
    if nonfinite is not None:
        warnings.append(nonfinite)
    warnings.extend(pc.warnings)
    result = _verify_result_dict(pc.report, args, compute_unit=pc.effective_unit)
    if pc.passed:
        return _Outcome(exit_code=0, result=result, warnings=warnings)
    return _Outcome(
        exit_code=1,
        result=result,
        warnings=warnings,
        error=_error(
            "precision_check_failed",
            "outputs exceeded tolerance",
            hint=_PRECISION_FAILED_HINT,
        ),
    )


_RUNNERS = {
    "convert": _run_convert,
    "inspect": _run_inspect,
    "verify": _run_verify,
    "schema": _run_schema,
}


def _augment_exception_outcome(
    command: str, args: argparse.Namespace, outcome: _Outcome
) -> None:
    """Append an onnxruntime_missing warning to *outcome* when appropriate.

    Legacy-parity note for convert errors: the pre-refactor CLI emitted it
    only once the model had loaded (so not for io_error/invalid_model_file)
    and only unless the user disabled both ORT steps.  Extracted so that the
    MCP server can call it too, ensuring both surfaces emit identical envelopes.
    """
    if (
        command == "convert"
        and not (args.no_validate and args.no_verify)
        and outcome.error is not None
        and outcome.error["code"] not in ("io_error", "invalid_model_file")
        and not _onnxruntime_available()
    ):
        outcome.warnings.append(_warning("onnxruntime_missing", _ORT_MISSING_MSG))


# ---------------------------------------------------------------------------
# shared dispatcher (the single entry point for both CLI and MCP)
# ---------------------------------------------------------------------------


def _execute_command(command: str, ns: argparse.Namespace) -> tuple[_Outcome, bool]:
    """Run *command*'s runner; map domain failures to structured errors.

    Returns ``(outcome, from_exception)``. ValueError/OverflowError are
    deliberately NOT caught: user-facing instances reach here wrapped in
    ConversionError already; what remains is internal bugs whose traceback
    is the bug report. The except tuple is the single definition of "domain
    failure" for BOTH the CLI and the MCP server.
    """
    try:
        return _RUNNERS[command](ns), False
    except (CoreaiOnnxError, OSError, ValidationError, DecodeError) as exc:
        outcome = _Outcome(exit_code=1, error=_error_from_exception(exc))
        _augment_exception_outcome(command, ns, outcome)
        return outcome, True
