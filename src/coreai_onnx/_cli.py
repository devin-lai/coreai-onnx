# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Command-line interface for coreai-onnx: convert, inspect, verify."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from ._service import _envelope, _execute_command, _Outcome

console = Console()


def _require_result(outcome: _Outcome) -> dict:
    if outcome.result is None:
        raise RuntimeError("renderer received no result payload")
    return outcome.result


def _require_error(outcome: _Outcome) -> dict:
    if outcome.error is None:
        raise RuntimeError("renderer received no error payload")
    return outcome.error


# ---------------------------------------------------------------------------
# human renderers (print, no computing) - output is byte-identical to the
# pre-refactor CLI; tests/test_cli.py is the no-regression suite.
# ---------------------------------------------------------------------------


def _print_verify_table(precision: dict) -> None:
    """Render a serialized verify result as a precision comparison table
    (shared by the convert and verify commands)."""
    table = Table(title="Precision: ONNX Runtime vs .aimodel", show_header=True)
    table.add_column("Output")
    table.add_column("Max abs error", justify="right")
    table.add_column("Max rel error", justify="right")
    table.add_column("PSNR", justify="right")
    table.add_column("Pass")

    for out in precision["outputs"]:
        pass_cell = "[green]PASS[/green]" if out["passed"] else "[red]FAIL[/red]"
        table.add_row(
            out["name"],
            _format_scientific(out["max_abs_error"]),
            _format_scientific(out["max_rel_error"]),
            _format_fixed(out["psnr"]),
            pass_cell,
        )

    console.print(table)


def _format_scientific(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.4e}"


def _format_fixed(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.2f}"


def _render_inspect(outcome: _Outcome, args: argparse.Namespace) -> None:
    r = _require_result(outcome)
    table = Table(title=f"Op histogram: {r['model_path']}", show_header=True)
    table.add_column("Op", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Status")

    for entry in r["ops"]:
        status = (
            "[green]supported[/green]"
            if entry["supported"]
            else "[red]unsupported[/red]"
        )
        table.add_row(entry["op"], str(entry["count"]), status)

    console.print(table)

    total_label = f"Total nodes: {r['total_nodes']}"
    if r["convertible"]:
        console.print(f"{total_label}  |  [bold green]convertible: yes[/bold green]")
    else:
        unsupported_ops = ", ".join(r["unsupported"])
        console.print(
            f"{total_label}  |  [bold red]convertible: no[/bold red]"
            f"  (unsupported: {unsupported_ops})"
        )


def _render_convert(outcome: _Outcome, args: argparse.Namespace) -> None:
    for w in outcome.warnings:
        if w["code"] == "onnxruntime_missing":
            console.print(f"[yellow]Note:[/yellow] {w['message']}")

    err = outcome.error
    if err is not None and err["code"] == "compiler_failed":
        console.print(f"[bold red]Error:[/bold red] {err['message']}")
        return

    r = _require_result(outcome)
    console.print(
        f"[bold green]Converted successfully[/bold green]\n"
        f"  Nodes: {r['total_nodes']}\n"
        f"  Output: {r['output_path']}"
    )

    for repair in r.get("repairs") or []:
        console.print(f"[yellow]Repaired:[/yellow] {repair['summary']}")

    for w in outcome.warnings:
        if w["code"] == "platform_no_runtime":
            console.print(f"[yellow]Note:[/yellow] {w['message']}")

    if err is not None and err["code"] == "precision_check_error":
        console.print(
            f"[bold red]Precision check could not run:[/bold red] {err['message']}"
        )
        return

    if r["precision"] is None:
        return
    _print_verify_table(r["precision"])
    if err is None:
        console.print("[bold green]Precision check passed[/bold green]")
    else:
        console.print(
            "[bold red]Precision check failed[/bold red] "
            "(outputs exceeded tolerance; the .aimodel was still written)"
        )


def _render_verify(outcome: _Outcome, args: argparse.Namespace) -> None:
    err = outcome.error
    if err is not None and outcome.result is None:
        if err["code"] == "platform_unsupported":
            console.print(f"[bold red]Error:[/bold red] {err['message']}")
        else:
            console.print(
                f"[bold red]Error during verification:[/bold red] {err['message']}"
            )
        return

    _print_verify_table(_require_result(outcome))
    if err is None:
        console.print("[bold green]Verification passed[/bold green]")
    else:
        console.print("[bold red]Verification failed[/bold red]")


def _render_schema(outcome: _Outcome, args: argparse.Namespace) -> None:
    r = _require_result(outcome)
    table = Table(title=f"coreai-onnx {r['tool']['version']}", show_header=True)
    table.add_column("Command", style="bold")
    table.add_column("Description")
    for cmd in r["commands"]:
        table.add_row(cmd["name"], cmd["summary"])
    console.print(table)
    console.print(
        f"Supported ops: {len(r['supported_ops'])}  |  "
        f"error codes: {len(r['error_codes'])}  |  "
        f"exit codes: {len(r['exit_codes'])}\n"
        "Run with --json for the full machine-readable contract."
    )


def _render_exception(outcome: _Outcome) -> None:
    err = _require_error(outcome)
    for w in outcome.warnings:
        console.print(f"[yellow]Note:[/yellow] {w['message']}")
    console.print(f"[bold red]Error:[/bold red] {err['message']}")


# ---------------------------------------------------------------------------
# parser + entry point
# ---------------------------------------------------------------------------


def _add_json_flag(p: argparse.ArgumentParser) -> None:
    # Also accept --json after the subcommand. default=SUPPRESS is load-bearing:
    # a plain default would overwrite the top-level flag already parsed from
    # `coreai-onnx --json <command>` (subparser defaults clobber the parent
    # namespace, bpo-9351).
    p.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai-onnx",
        description="Convert, inspect, and verify ONNX models for Core AI.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit one machine-readable JSON document on stdout instead of "
        "human-formatted output",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # convert
    p_convert = sub.add_parser(
        "convert",
        help="Convert an ONNX model to .aimodel",
        description="Convert an ONNX model to .aimodel",
    )
    p_convert.add_argument("model", help="Path to the .onnx model file")
    p_convert.add_argument("-o", "--output", required=True, help="Output .aimodel path")
    p_convert.add_argument(
        "--no-optimize",
        action="store_true",
        default=False,
        help="Skip the optimize() step after conversion",
    )
    p_convert.add_argument(
        "--name",
        default="main",
        help="Entrypoint function name (default: main)",
    )
    p_convert.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip the pre-conversion ONNX Runtime validation gate",
    )
    p_convert.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="Skip the post-conversion precision check (ONNX Runtime vs .aimodel)",
    )
    p_convert.add_argument(
        "--repair",
        action="store_true",
        default=False,
        help="Automatically apply known-safe ONNX rewrites for documented Core "
        "AI runtime limitations (e.g. promote float16 programs to float32), "
        "then re-verify parity; applied fixes are listed in result.repairs",
    )
    p_convert.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Precision-check relative tolerance (default: 1e-3, or 1e-2 for float16)",
    )
    p_convert.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Precision-check absolute tolerance (default: 1e-4, or 1e-3 for float16)",
    )
    p_convert.add_argument(
        "--min-psnr",
        type=float,
        default=None,
        help="Also accept an output that fails rtol/atol but reaches this "
        "PSNR in dB (for benign accumulation noise on large-magnitude "
        "outputs; unset by default)",
    )
    p_convert.add_argument(
        "--compute-unit",
        choices=["cpu_only", "cpu", "gpu", "ane"],
        default=None,
        help="Compute units the precision check may execute on (default: "
        "runtime's choice; cpu_only is the fp32 path)",
    )
    p_convert.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for validation/precision input generation (default: 0)",
    )
    _add_json_flag(p_convert)

    # inspect
    p_inspect = sub.add_parser(
        "inspect",
        help="Show op coverage for a model",
        description="Show op coverage for a model",
    )
    p_inspect.add_argument("model", help="Path to the .onnx model file")
    _add_json_flag(p_inspect)

    # verify
    p_verify = sub.add_parser(
        "verify",
        help="Verify .aimodel output against onnxruntime",
        description="Verify .aimodel output against onnxruntime",
    )
    p_verify.add_argument("model", help="Path to the .onnx model file")
    p_verify.add_argument("aimodel", help="Path to the .aimodel bundle")
    p_verify.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance (default: 1e-3, or 1e-2 for float16 outputs)",
    )
    p_verify.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance (default: 1e-4, or 1e-3 for float16 outputs)",
    )
    p_verify.add_argument(
        "--min-psnr",
        type=float,
        default=None,
        help="Also accept an output that fails rtol/atol but reaches this "
        "PSNR in dB (for benign accumulation noise on large-magnitude "
        "outputs; unset by default)",
    )
    p_verify.add_argument(
        "--compute-unit",
        choices=["cpu_only", "cpu", "gpu", "ane"],
        default=None,
        help="Compute units the check may execute on (default: runtime's "
        "choice; cpu_only is the fp32 path)",
    )
    p_verify.add_argument(
        "--seed", type=int, default=0, help="RNG seed for input generation"
    )
    p_verify.add_argument(
        "--name",
        default="main",
        help="Entrypoint function name used at convert time (default: main)",
    )
    _add_json_flag(p_verify)

    # schema
    p_schema = sub.add_parser(
        "schema",
        help="Print the tool's machine-readable capability description",
        description="Print the tool's machine-readable capability description",
    )
    _add_json_flag(p_schema)

    return parser


_RENDERERS = {
    "convert": _render_convert,
    "inspect": _render_inspect,
    "verify": _render_verify,
    "schema": _render_schema,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    outcome, from_exception = _execute_command(args.command, args)

    if args.json:
        print(json.dumps(_envelope(args.command, outcome), indent=2, allow_nan=False))
    elif from_exception:
        _render_exception(outcome)
    else:
        _RENDERERS[args.command](outcome, args)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
