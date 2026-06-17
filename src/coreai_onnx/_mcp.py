# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MCP server for coreai-onnx (optional [mcp] extra).

Exposes inspect/convert/verify/schema as MCP tools. Each tool wraps the
corresponding command runner in coreai_onnx._service and returns the SAME envelope
the CLI emits with --json (schema_version 1; see docs/cli.md), so an agent's
knowledge of the CLI contract transfers unchanged. Domain failures come back
as status:"error" envelopes — never as MCP protocol errors. Exit codes do not
exist over MCP; status/error.code carry the same information.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from . import _service

_MCP_HINT = 'the MCP server requires the [mcp] extra: pip install "coreai-onnx[mcp]"'

_INSTRUCTIONS = (
    "Run coreai-onnx conversion, inspection, verification, and schema commands. "
    "All tools return the "
    "same JSON envelope as the coreai-onnx CLI's --json mode: "
    "{schema_version, command, status, result, warnings, error}. Branch on "
    "status and error.code (stable, append-only); see AGENTS.md in the "
    "coreai-onnx repository for the full playbook. Start with inspect_model "
    "to check op coverage, then convert_model; a precision_check_failed "
    "error still means the .aimodel was written."
)


async def _call_runner(command: str, ns: argparse.Namespace) -> dict[str, Any]:
    """Run a command in a worker thread and wrap the outcome in an envelope.

    The thread is load-bearing twice: the runners block (conversion can take
    a while), and _run_convert/_run_verify call asyncio.run internally, which
    raises if invoked on the MCP server's own event-loop thread.
    """
    # Imported lazily (like FastMCP in _build_server) so this module stays
    # importable without the [mcp] extra; anyio ships with mcp.
    import anyio

    def work() -> dict[str, Any]:
        outcome, _ = _service._execute_command(command, ns)
        return _service._envelope(command, outcome)

    return await anyio.to_thread.run_sync(work)


def _build_server():
    """Create the FastMCP server. Raises ImportError with an install hint
    when the [mcp] extra is missing (mcp is imported lazily so the rest of
    coreai_onnx never depends on it)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(_MCP_HINT) from exc

    server = FastMCP("coreai-onnx", instructions=_INSTRUCTIONS)

    @server.tool(structured_output=True)
    async def inspect_model(model_path: str) -> dict[str, Any]:
        """Analyze an ONNX model's op coverage before converting.

        Returns the coreai-onnx envelope; result.convertible says whether
        every op has a Core AI lowering, result.unsupported lists the ones
        that do not. A non-convertible model is status:"ok" — it is a
        result, not an error.
        """
        return await _call_runner("inspect", argparse.Namespace(model=model_path))

    @server.tool(structured_output=True)
    async def convert_model(
        model_path: str,
        output_path: str,
        optimize: bool = True,
        validate: bool = True,
        verify: bool = True,
        repair: bool = False,
        rtol: float | None = None,
        atol: float | None = None,
        min_psnr: float | None = None,
        compute_unit: str | None = None,
        seed: int = 0,
        entrypoint: str = "main",
    ) -> dict[str, Any]:
        """Convert an ONNX model to a Core AI .aimodel bundle.

        Validates the model on ONNX Runtime first (validate=True), converts,
        optimizes (optimize=True), saves to output_path, then compares the
        .aimodel against ONNX Runtime (verify=True; needs macOS 27+). Returns
        the coreai-onnx envelope. With repair=True, documented Core AI runtime
        limitations are fixed automatically with known-safe ONNX rewrites (e.g.
        float16 programs are promoted to float32) before conversion, and the
        applied fixes are listed in result.repairs; the precision check is the
        safety net that rejects any rewrite that would change results. On
        error.code "precision_check_failed" the .aimodel WAS written — inspect
        result.precision instead of treating it as fatal; on
        "precision_check_error" the .aimodel was also written and the
        conversion is unaffected, but the check could not complete
        (result.precision is null — see error.message). This includes the
        native runtime aborting while executing the program on the selected
        compute unit; that abort is contained (it no longer kills this server),
        and a plain retry would only repeat it — re-check with
        compute_unit="cpu_only" (the fp32 fidelity path) instead. rtol/atol of
        null use per-dtype defaults. By default the verdict measures fp32
        conversion fidelity: the check runs on the runtime's fast default unit
        and falls back to cpu_only only on a miss, so float16 GPU/ANE divergence
        (warning "precision_hardware_divergence") and benign high-PSNR
        accumulation noise (warning "precision_benign_noise"; floor = min_psnr,
        default 50 dB) pass instead of failing. result.precision.compute_unit is
        the effective unit (null = default unit, "cpu_only" = fp32 fallback, or
        the pinned unit). Pin compute_unit to "cpu_only"/"cpu"/"gpu"/"ane" to
        make that unit the verdict with no fallback (GPU/ANE run float16).
        """
        # The Namespaces below mirror _cli._build_parser()'s dests AND
        # defaults (name="main", seed=0, repair=False,
        # rtol/atol/min_psnr/compute_unit=None); keep them in sync.
        ns = argparse.Namespace(
            model=model_path,
            output=output_path,
            no_optimize=not optimize,
            name=entrypoint,
            no_validate=not validate,
            no_verify=not verify,
            repair=repair,
            rtol=rtol,
            atol=atol,
            min_psnr=min_psnr,
            compute_unit=compute_unit,
            seed=seed,
        )
        return await _call_runner("convert", ns)

    @server.tool(structured_output=True)
    async def verify_model(
        model_path: str,
        aimodel_path: str,
        rtol: float | None = None,
        atol: float | None = None,
        min_psnr: float | None = None,
        compute_unit: str | None = None,
        seed: int = 0,
        entrypoint: str = "main",
    ) -> dict[str, Any]:
        """Re-check an existing .aimodel's numerical parity vs ONNX Runtime.

        Requires macOS 27+ with the Core AI runtime (error.code
        "platform_unsupported" otherwise). Returns the coreai-onnx envelope;
        result.outputs carries per-output max abs/rel error and PSNR (PSNR of
        infinity serializes as the string "inf"). The verdict follows the same
        default-fidelity policy as convert: with no compute_unit it measures the
        fp32 path (default unit, cpu_only fallback on a miss), demoting float16
        divergence and benign high-PSNR noise to the warnings
        "precision_hardware_divergence"/"precision_benign_noise"; min_psnr is the
        benign floor (default 50 dB). result.compute_unit is the effective unit
        (null = default unit, "cpu_only" = fp32 fallback, or the pinned unit).
        Pin compute_unit to "cpu_only"/"cpu"/"gpu"/"ane" for that unit with no
        fallback (GPU/ANE run float16).
        """
        ns = argparse.Namespace(
            model=model_path,
            aimodel=aimodel_path,
            rtol=rtol,
            atol=atol,
            min_psnr=min_psnr,
            compute_unit=compute_unit,
            seed=seed,
            name=entrypoint,
        )
        return await _call_runner("verify", ns)

    @server.tool(structured_output=True)
    async def get_schema() -> dict[str, Any]:
        """Get coreai-onnx's machine-readable capability contract.

        Same payload as `coreai-onnx schema --json`: commands and flags, all
        error/warning/exit codes with meanings, the supported-op list, and
        runtime requirements.
        """
        return await _call_runner("schema", argparse.Namespace())

    return server


def main() -> int:
    """Console-script entry point: serve MCP over stdio.

    Stdout belongs to the protocol; this function must never print to it.
    """
    try:
        server = _build_server()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    server.run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
