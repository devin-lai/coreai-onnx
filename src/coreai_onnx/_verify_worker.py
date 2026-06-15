# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Out-of-process execution of a saved .aimodel, for crash isolation.

Running a converted program enters the native Core AI runtime, which can
*abort the process* (call ``abort()``, not raise) when it cannot execute a
program on the selected compute unit - e.g. the MPSGraph ``IOSurface``
assertion seen on the GPU path. A one-shot CLI invocation merely loses its JSON
envelope to that; the long-lived MCP server loses the whole process. Neither is
acceptable for a tool with a stable error contract.

So the single native step (``coreai_onnx._verify._execute_asset``) runs in a
child process here. A clean run returns the outputs; a child killed by a signal,
or one that exits non-zero, is turned into a ``CoreaiOnnxError`` - which the
service layer already maps to ``precision_check_error`` (convert exit 3, verify
exit 1). No new error code, no new exit code: a failure class that used to
escape the contract now lands inside it.

Wire protocol (parent <-> child, through one private temp dir):

==============  =========  ============================================
file            direction  contents
==============  =========  ============================================
``meta.json``   parent->   aimodel path, I/O names, entrypoint, unit
``inputs.npz``  parent->   feed arrays, positional (names are in meta)
``outputs.npz`` ->parent   result arrays, positional (names in meta)
``error.json``  ->parent   a *handled* child exception ``{type, message}``
==============  =========  ============================================

Arrays travel positionally because ONNX tensor names ("/enc/Add_output_0")
are not valid ``np.savez`` keyword keys; the ordered name lists in ``meta.json``
re-key them. The child writes ``outputs.npz`` XOR ``error.json`` and exits 0 XOR
1; any other outcome (signal death, non-zero exit, missing files) is a crash.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .errors import CoreaiOnnxError

_WORKER_MODULE = "coreai_onnx._verify_worker"


def run_asset_isolated(
    aimodel_path: str | Path,
    inputs: dict[str, np.ndarray],
    asset_output_names: Sequence[str],
    entrypoint: str,
    compute_unit: str | None,
) -> dict[str, np.ndarray]:
    """Run the .aimodel in a child process; return ``{output_name: array}``.

    Raises ``CoreaiOnnxError`` if the child crashes (native abort / signal /
    non-zero exit) or reports a handled exception, so the caller's existing
    ``except`` path turns it into a structured ``precision_check_error``.
    """
    output_names = list(asset_output_names)
    input_names = list(inputs)
    with tempfile.TemporaryDirectory(prefix="coreai_onnx_verify_") as td:
        workdir = Path(td)
        np.savez(workdir / "inputs.npz", *(np.asarray(inputs[n]) for n in input_names))
        (workdir / "meta.json").write_text(
            json.dumps(
                {
                    "aimodel_path": str(aimodel_path),
                    "input_names": input_names,
                    "output_names": output_names,
                    "entrypoint": entrypoint,
                    "compute_unit": compute_unit,
                }
            )
        )
        proc = subprocess.run(
            [sys.executable, "-m", _WORKER_MODULE, str(workdir)],
            capture_output=True,
            text=True,
            check=False,
        )
        return _interpret(proc.returncode, proc.stderr, workdir, output_names)


def _interpret(
    returncode: int, stderr: str, workdir: Path, output_names: list[str]
) -> dict[str, np.ndarray]:
    """Map a finished child process to outputs or a ``CoreaiOnnxError``.

    Pure and side-effect free given the temp dir, so the crash-handling
    contract is unit-testable without the native runtime.
    """
    outputs_file = workdir / "outputs.npz"
    error_file = workdir / "error.json"
    if returncode == 0 and outputs_file.exists():
        with np.load(outputs_file) as data:
            return {n: np.array(data[f"arr_{i}"]) for i, n in enumerate(output_names)}
    if error_file.exists():
        info = json.loads(error_file.read_text())
        raise CoreaiOnnxError(
            f"running the .aimodel raised "
            f"{info.get('type', 'Error')}: {info.get('message', '')}"
        )
    raise CoreaiOnnxError(_crash_message(returncode, stderr))


def _crash_message(returncode: int, stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    detail = f" ({lines[-1].strip()})" if lines else ""
    if returncode < 0:
        try:
            signame = signal.Signals(-returncode).name
        except ValueError:
            signame = f"signal {-returncode}"
        cause = f"the Core AI runtime aborted (killed by {signame})"
    else:
        cause = f"the Core AI runtime exited abnormally (exit code {returncode})"
    return (
        f"{cause} while executing the .aimodel{detail}. The conversion itself is "
        "unaffected and the .aimodel was written; the runtime could not execute "
        "the program on the selected compute unit - re-check with "
        "--compute-unit cpu_only (the fp32 path that proves conversion fidelity)."
    )


def _worker_main(argv: list[str]) -> int:
    """Child entry point: execute the asset, then write outputs or error.json."""
    workdir = Path(argv[0])
    meta = json.loads((workdir / "meta.json").read_text())
    try:
        with np.load(workdir / "inputs.npz") as data:
            inputs = {
                n: np.array(data[f"arr_{i}"])
                for i, n in enumerate(meta["input_names"])
            }
        # Imported here (not at module top) so the parent side stays importable
        # without the native runtime, and to break the _verify <-> worker cycle.
        from ._verify import _execute_asset

        got = asyncio.run(
            _execute_asset(
                meta["aimodel_path"],
                inputs,
                meta["output_names"],
                meta["entrypoint"],
                meta["compute_unit"],
            )
        )
        np.savez(
            workdir / "outputs.npz",
            *(np.asarray(got[n]) for n in meta["output_names"]),
        )
        return 0
    except Exception as exc:
        # Report any handled error to the parent as error.json (exit 1). A
        # native abort is not catchable here - it kills this child, and the
        # parent infers it from the non-zero/signal exit.
        (workdir / "error.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)})
        )
        return 1


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
