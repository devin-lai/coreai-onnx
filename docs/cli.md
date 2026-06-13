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
- `--min-psnr` — also accept an output that fails `rtol`/`atol` but reaches this
  PSNR in dB (unset by default; see
  [Benign precision-check failures](#benign-precision-check-failures))
- `--compute-unit` — compute units the precision check may execute on
  (`cpu_only`, `cpu`, `gpu`, `ane`; default: the runtime's choice). GPU and ANE
  execute in float16; `cpu_only` is the fp32 path that proves conversion
  fidelity
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
- `--min-psnr` — also accept an output that fails `rtol`/`atol` but reaches this
  PSNR in dB (unset by default)
- `--compute-unit` — compute units the check may execute on (`cpu_only`,
  `cpu`, `gpu`, `ane`; default: the runtime's choice)
- `--seed` — RNG seed for input generation (default: `0`)
- `--name` — entrypoint function name used at convert time (default: `main`)

Requires `onnxruntime` and the Core AI runtime (macOS 27+).

(benign-precision-check-failures)=
### Benign precision-check failures

The default pass criterion is elementwise (`numpy.allclose`). Two classes of
correct conversions fail it, and the report tells them apart:

- **Large-magnitude accumulation noise.** When an output's values span many
  orders of magnitude (some vision encoders reach `1e12` on random probe
  inputs), even fp32-vs-fp32 summation-order differences produce absolute
  errors no sensible `rtol`/`atol` accepts elementwise — while PSNR stays
  excellent (> 100 dB). That is what `--min-psnr` is for: `--min-psnr 60`
  accepts an output whose signal-to-noise ratio is at least 60 dB even where
  the elementwise check fails. As reference points, 60 dB is a faithful
  conversion; 100+ dB is numerically near-perfect.
- **float16 hardware.** The runtime executes GPU/ANE in float16. A model
  whose intermediate values exceed float16 range (or whose divisions flush
  denormal denominators to zero) can drift or produce NaN there even though
  the conversion is exact — `--compute-unit cpu_only` checks the fp32 CPU
  path, which isolates conversion fidelity from float16 hardware behavior.
  If `cpu_only` passes and the default check NaNs, the model needs fp32:
  ship it with `SpecializationOptions.cpu_only()` (or fix the numerics in
  the source model, e.g. add an epsilon to unguarded divisions).

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
      "version": "1.0.0",
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
| `precision_check_error` | The precision check could not run; `details.exception_type` names the exception. |
| `invalid_model_file` | The file is not a valid ONNX model. |
| `io_error` | The file could not be read or written. |
| `platform_unsupported` | `verify` requires macOS 27+ with the Core AI runtime. |

Warning codes: `onnxruntime_missing` (onnxruntime not installed; validation
and precision check skipped), `platform_no_runtime` (not on macOS; precision
check skipped but `.aimodel` was written), `reference_nonfinite` (the ONNX
Runtime reference itself contains NaN/Inf on the probe input; parity at those
positions is checked by mask — see
[Benign precision-check failures](#benign-precision-check-failures)).

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
