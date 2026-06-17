# CLI Reference

The `coreai-onnx` command provides four subcommands.

## inspect

Analyse an ONNX model and report op coverage.

```
coreai-onnx inspect <model.onnx>
```

**Sample output** (single-Relu model):

```
  Op histogram: model.onnx
┏━━━━━━┳━━━━━━━┳━━━━━━━━━━━┓
┃ Op   ┃ Count ┃ Status    ┃
┡━━━━━━╇━━━━━━━╇━━━━━━━━━━━┩
│ Relu │     1 │ supported │
└──────┴───────┴───────────┘
Total nodes: 1  |  convertible: yes
```

Exit codes: see [Exit codes (stable, all commands)](#exit-codes-stable-all-commands). Exit `1` means the model contains unsupported ops (listed in the output).

## convert

Convert an ONNX model to a Core AI `.aimodel` asset. By default `convert` runs a
three-step pipeline:

1. **Validate** — confirm the input model loads and runs on ONNX Runtime, and
   stop with a clear error if it does not. Skipped, with a note, when
   `onnxruntime` is not installed.
2. **Convert** — lower the model and write the optimised `.aimodel`.
3. **Precision check** — run the saved `.aimodel` and print a precision
   comparison against ONNX Runtime. Skipped, with a note, off macOS or when
   `onnxruntime` is not installed.

ONNX Runtime is executed only once: the reference outputs from the validation
step are reused for the precision check. The reference session runs with ONNX
Runtime's graph optimizations and Arm KleidiAI kernels disabled so it reflects
ONNX spec semantics — both have produced wrong references on Apple Silicon
(ORT 1.24–1.26) — so reference outputs may differ slightly from a default ORT
session.

```
coreai-onnx convert <model.onnx> -o <output.aimodel> [options]
```

Options:

- `-o`, `--output` — output `.aimodel` path (required)
- `--no-optimize` — skip the `optimize()` step after conversion (optimisation runs
  by default)
- `--name` — entrypoint function name (default: `main`)
- `--no-validate` — skip the pre-conversion ONNX Runtime validation gate
- `--no-verify` — skip the post-conversion precision check
- `--repair` — apply known-safe ONNX rewrites for documented Core AI runtime
  limitations (e.g. promote float16 programs to float32) before conversion,
  then re-verify parity; applied fixes appear in `result.repairs` (see
  [repair](repair.md))
- `--rtol` — precision-check relative tolerance (default: `1e-3`, or `1e-2` for float16 outputs)
- `--atol` — precision-check absolute tolerance (default: `1e-4`, or `1e-3` for float16 outputs)
- `--min-psnr` — PSNR (dB) floor below which a `rtol`/`atol` failure is *not*
  accepted as benign accumulation noise (default: `50`; see
  [The default precision verdict](#the-default-precision-verdict) and
  [Benign precision-check failures](#benign-precision-check-failures))
- `--compute-unit` — pin the precision check to a unit (`cpu_only`, `cpu`,
  `gpu`, `ane`). Default: the verdict measures fp32 fidelity (the check runs on
  the runtime's choice and falls back to `cpu_only` only on a miss). GPU and ANE
  execute in float16; `cpu_only` is the fp32 path that proves conversion
  fidelity. An explicit value is the verdict, with no fallback
- `--seed` — RNG seed for input generation (default: `0`)

**Sample output:**

```
Converted successfully
  Nodes: 1
  Output: model.aimodel
           Precision: ONNX Runtime vs .aimodel
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Output ┃ Max abs error ┃ Max rel error ┃   PSNR ┃ Pass ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ y      │    5.9605e-08 │    1.1217e-07 │ 145.59 │ PASS │
└────────┴───────────────┴───────────────┴────────┴──────┘
Precision check passed
```

Exit codes: see [Exit codes (stable, all commands)](#exit-codes-stable-all-commands). Exit `3` means the `.aimodel` was written but the precision check did not pass (exceeded tolerance, or could not run); exit `1` means conversion failed and no `.aimodel` was written.

> **Validation inputs:** the validate and precision steps feed seeded random
> tensors. For models with index-style inputs (e.g. gather indices) those values
> can fall out of range and trip a false validation failure — pass
> `--no-validate` (and `--no-verify`) for such models, or use the Python API
> (`validate_onnxruntime` / `verify`) with explicit `inputs`.

## verify

Run the converted model on random inputs and compare outputs against ONNX Runtime.

```
coreai-onnx verify <model.onnx> <model.aimodel> [--rtol RTOL] [--atol ATOL] [--min-psnr DB] [--compute-unit UNIT] [--seed 0] [--name ENTRYPOINT]
```

Options:

- `--rtol` — relative tolerance (default: `1e-3`, or `1e-2` for float16 outputs)
- `--atol` — absolute tolerance (default: `1e-4`, or `1e-3` for float16 outputs)
- `--min-psnr` — PSNR (dB) floor below which a `rtol`/`atol` failure is *not*
  accepted as benign accumulation noise (default: `50`)
- `--compute-unit` — pin the check to a unit (`cpu_only`, `cpu`, `gpu`, `ane`).
  Default: the verdict measures fp32 fidelity (runs on the runtime's choice,
  falls back to `cpu_only` only on a miss; an explicit value has no fallback)
- `--seed` — RNG seed for input generation (default: `0`)
- `--name` — entrypoint function name used at convert time (default: `main`)

Requires `onnxruntime` and the Core AI runtime (macOS 27+).

(the-default-precision-verdict)=
### The default precision verdict

The precision check answers one question — *did the conversion preserve the
model?* — which is a property of the **fp32** computation, not of any one
hardware unit. So when you do not pass `--compute-unit`, the verdict measures
conversion fidelity, and two correct-but-divergent cases that used to fail are
demoted to warnings:

- **float16 hardware divergence.** The runtime executes GPU/ANE in float16, so a
  model whose intermediates exceed float16 range (or whose program the GPU
  backend cannot execute at all) can drift, NaN, or abort there while the
  conversion is exact. When the default-unit run fails or the runtime aborts,
  the check re-runs on the deterministic fp32 `cpu_only` path; if that passes,
  the verdict is **pass** with a `precision_hardware_divergence` warning.
- **Benign accumulation noise.** An output that fails the elementwise tolerance
  but reaches a high PSNR (≥ 50 dB by default; see below) and matches the
  reference's non-finite pattern is accepted with a `precision_benign_noise`
  warning.

For speed, the check runs first on the runtime's default unit (the fast path);
the fp32 `cpu_only` confirmation runs **only** when that first run does not pass,
so a passing conversion pays no extra cost. `precision.compute_unit` in the
envelope is the *effective* unit behind the verdict: `null` when the default
unit was authoritative, `"cpu_only"` when the fp32 confirmation was, or the unit
you pinned. An explicit `--compute-unit` is honored exactly, with no fallback —
pin `gpu`/`ane` to make the float16 hardware path the verdict, or `cpu_only` to
force the fp32 path.

(benign-precision-check-failures)=
### Benign precision-check failures

The strict per-element criterion is `numpy.allclose`. Two classes of correct
conversions fail it elementwise; by default
([The default precision verdict](#the-default-precision-verdict)) both pass with
a warning rather than failing, and the report tells them apart:

- **Large-magnitude accumulation noise.** When an output's values span many
  orders of magnitude (some vision encoders reach `1e12` on random probe
  inputs), even fp32-vs-fp32 summation-order differences produce absolute
  errors no sensible `rtol`/`atol` accepts elementwise — while PSNR stays
  excellent (> 100 dB). The default `--min-psnr 50` floor accepts these as
  benign (with a `precision_benign_noise` warning); raise the floor, or set
  `--rtol`/`--atol`, to treat them as failures. As reference points, 60 dB is a
  faithful conversion; 100+ dB is numerically near-perfect.
- **float16 hardware.** The runtime executes GPU/ANE in float16. A model
  whose intermediate values exceed float16 range (or whose divisions flush
  denormal denominators to zero) can drift or produce NaN there even though
  the conversion is exact. The default verdict already isolates this: it falls
  back to the fp32 `cpu_only` path and passes with a
  `precision_hardware_divergence` warning. If the model needs fp32 at runtime,
  ship it with `SpecializationOptions.cpu_only()` (or fix the numerics in the
  source model, e.g. add an epsilon to unguarded divisions); pin
  `--compute-unit gpu`/`ane` to make the float16 path the verdict.

A third case is the model itself producing NaN/Inf on the random probe input
(e.g. an unguarded `0/0`). The reference and the converted model then both
emit non-finite values; the check compares those positions by NaN/Inf mask,
reports per-output counts as `expected_nonfinite`, and emits the
`reference_nonfinite` warning so the degenerate reference is visible.

**Sample output:**

```
           Precision: ONNX Runtime vs .aimodel
┏━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━┓
┃ Output ┃ Max abs error ┃ Max rel error ┃ PSNR ┃ Pass ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━┩
│ y      │    0.0000e+00 │    0.0000e+00 │  inf │ PASS │
└────────┴───────────────┴───────────────┴──────┴──────┘
Verification passed
```

Exit codes: see [Exit codes (stable, all commands)](#exit-codes-stable-all-commands). Exit `2` means the Core AI runtime is unavailable (requires macOS 27+).

## schema

Print the tool's machine-readable capability description.

```
coreai-onnx schema [--json]
```

In human mode, prints a summary table of commands with a count of supported
ops, error codes, and exit codes.

With `--json`, emits the full capability contract — commands with all flags,
error codes, warning codes, exit codes, the supported-op list, and runtime
requirements. The code tables in the dump are the same source the CLI emits
from, so they cannot drift from behavior.

Exit codes: always `0` (the command needs no model, no optional dependencies,
and works on any platform).

**`--json` example** (trimmed — `commands` and `supported_ops` bodies omitted):

```text
{
  "schema_version": 1,
  "command": "schema",
  "status": "ok",
  "result": {
    "tool": {
      "name": "coreai-onnx",
      "version": "1.1.1",
      "schema_version": 1,
      "description": "Convert ONNX models to Apple Core AI (.aimodel) — the AI-first successor to Core ML on iOS 27. Validated, precision-checked, agent-friendly, with automatic float16 repair and an MCP server.",
      "global_options": [
        {
          "flag": "--json",
          "type": "flag",
          "default": false,
          "required": false,
          "help": "Emit one machine-readable JSON document on stdout instead of human-formatted output"
        }
      ]
    },
    "commands": [ ... ],
    "error_codes": [ ... ],
    "warning_codes": [ ... ],
    "exit_codes": [ ... ],
    "supported_ops": [ ... ],
    "runtime": {
      "convert_platforms": "any",
      "verify_platforms": "macOS 27+ with the Core AI runtime",
      "onnxruntime_required_for": ["validate", "verify"]
    }
  },
  "warnings": [],
  "error": null
}
```

## Machine-readable output (`--json`)

Every command accepts `--json` (before or after the subcommand). Stdout then
carries exactly one JSON document — nothing else — in this envelope:

```json
{
  "schema_version": 1,
  "command": "convert",
  "status": "ok",
  "result": { },
  "warnings": [{"code": "onnxruntime_missing", "message": "..."}],
  "error": null
}
```

On failure, `status` is `"error"` and `error` is an object with keys
`code`, `message`, `details`, and `hint`. `result` may still be present when
partial output exists (for example, the `.aimodel` was written but the
precision check failed). Non-finite precision metrics serialize as strings:
PSNR of infinity as `"inf"`, negative infinity (zero reference signal,
nonzero noise) as `"-inf"`, and NaN/Inf error maxima as `"nan"`/`"inf"`.
`rtol`/`atol`/`min_psnr`/`compute_unit` in verify results echo the values
passed on the command line; `null` means the default was chosen inside
`verify()`.

### Command results

- **`inspect`** — `{model_path, total_nodes, convertible, ops: [{op, count,
  supported}], unsupported: [...]}` (ops sorted by name). Exit code `1` with
  `status: "ok"` means the analysis succeeded but the model is not
  convertible.
- **`convert`** — `{output_path, total_nodes, optimized, validated, repairs,
  precision}`; `repairs` lists any auto-repairs applied with `--repair`
  (empty otherwise; each is `{name, summary, details}` — see
  [repair](repair.md)); `precision` is `null` when the check was skipped —
  either explicitly (`--no-verify`) or for a reason reported in `warnings`.
- **`verify`** — `{passed, rtol, atol, min_psnr, compute_unit, seed,
  outputs: [{name, max_abs_error, max_rel_error, psnr, passed,
  expected_nonfinite}]}` (outputs in model order; `expected_nonfinite` counts
  NaN/Inf values in the ONNX Runtime reference for that output).

### Error codes (stable, append-only)

| Code | Meaning |
| --- | --- |
| `unsupported_ops` | Ops with no lowering; `details.missing` is a dict mapping op key to a list of example node names. |
| `model_validation_failed` | The input model failed to load or run on ONNX Runtime. |
| `conversion_failed` | A lowering failed; `details` carries `node_name` and `op_key` when known. |
| `compiler_failed` | The Core AI compiler failed to optimize or save the program. |
| `precision_check_failed` | Outputs exceeded tolerance (asset still written for `convert`). |
| `precision_check_error` | The precision check could not run; `details.exception_type` names the exception. Includes the native Core AI runtime aborting while executing the `.aimodel` on the selected compute unit (the check runs out-of-process, so that abort is contained, not a process crash) — the `.aimodel` was still written and the conversion is unaffected; re-check with `--compute-unit cpu_only`. |
| `invalid_model_file` | The file is not a valid ONNX model. |
| `io_error` | The file could not be read or written. |
| `platform_unsupported` | `verify` requires macOS 27+ with the Core AI runtime. |

Warning codes: `onnxruntime_missing` (onnxruntime not installed; validation
and precision check skipped), `platform_no_runtime` (not on macOS; precision
check skipped but `.aimodel` was written), `reference_nonfinite` (the ONNX
Runtime reference itself contains NaN/Inf on the probe input; parity at those
positions is checked by mask — see
[Benign precision-check failures](#benign-precision-check-failures)),
`precision_benign_noise` (an output failed the elementwise tolerance but was
accepted as benign accumulation noise at high PSNR — the conversion is
faithful), `precision_hardware_divergence` (the model diverged on the runtime's
default float16 GPU/ANE unit but is faithful on the fp32 CPU path; the verdict
reflects fp32 — see [The default precision verdict](#the-default-precision-verdict)).

(exit-codes-stable-all-commands)=
### Exit codes (stable, all commands)

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Failure: bad model, unsupported ops, conversion failure, or verification failure. |
| `2` | Platform error (`verify` on non-macOS) or usage error (argparse, reported on stderr). |
| `3` | The `.aimodel` was written but the precision check failed or could not run (`convert` only). |

### Stability policy

`schema_version` is `1` and bumps only on breaking changes; additive fields
do not bump it. Error codes, warning codes, and exit codes are append-only.
