# coreai-onnx — Agent Guide

How to drive coreai-onnx programmatically. Every command accepts `--json` and
prints exactly one JSON envelope on stdout; exit codes and error codes are
stable contracts (see Stability, below).

## Install

| Goal | Command | Platforms |
|---|---|---|
| Convert only | `pip install coreai-onnx` | any |
| + validation & precision checks | `pip install "coreai-onnx[verify]"` | any (the precision check itself needs macOS 27+) |

Executing a converted `.aimodel` requires macOS 27+ / iOS 27+ (Core AI
framework). Conversion itself runs anywhere.

## Discover capabilities

    coreai-onnx schema --json

Returns the full machine-readable contract: commands, flags, error codes,
warning codes, exit codes, the supported-op list, and runtime requirements.
Treat this as the source of truth — it is generated from the same tables the
CLI emits from, so it cannot drift from behavior.

## MCP server

Prefer native tool calls? `pip install "coreai-onnx[mcp]"` and register the
stdio server `coreai-onnx-mcp` with your MCP client:

```json
{"mcpServers": {"coreai-onnx": {"command": "coreai-onnx-mcp"}}}
```

It exposes `inspect_model`, `convert_model`, `verify_model`, and
`get_schema`, each returning the same envelope documented here — branch on
`status`/`error.code` exactly as below. Exit codes do not exist over MCP.
Boolean parameters replace the CLI's negative flags (`optimize=false` ≙
`--no-optimize`); `entrypoint` ≙ `--name`.

## The envelope

Every `--json` invocation prints exactly one object of this shape on stdout:

```json
{
  "schema_version": 1,
  "command": "<convert|inspect|verify|schema>",
  "status": "<ok|error>",
  "result": { "...": "command-specific payload, or null" },
  "warnings": [ { "code": "...", "message": "..." } ],
  "error": { "code": "...", "message": "...", "details": {}, "hint": "..." }
}
```

Rules:

- `status` is `"error"` if and only if `error` is non-null.
- `result` may be partial or null on error — read what is present, do not
  require every key.
- Non-finite precision metrics serialize as strings — `"inf"`, `"-inf"`, or
  `"nan"` (JSON has no literals for them).
- `warnings` is always an array (possibly empty); each entry has a stable
  `code` and a human-readable `message`.

## Canonical workflow

The examples below are real, unedited CLI output (a single-`Relu` model and a
single-`Det` model, converted in a temp directory on macOS with onnxruntime
installed).

1. **Check coverage first** — `coreai-onnx inspect model.onnx --json`

   ```json
   {
     "schema_version": 1,
     "command": "inspect",
     "status": "ok",
     "result": {
       "model_path": "/var/folders/f9/k__xp4rn4_97h3q7p8597k840000gn/T/tmp6ccnaupb/relu.onnx",
       "total_nodes": 1,
       "convertible": true,
       "ops": [
         {
           "op": "Relu",
           "count": 1,
           "supported": true
         }
       ],
       "unsupported": []
     },
     "warnings": [],
     "error": null
   }
   ```

   Gate on `result.convertible`. Exit code 1 with `status: "ok"` means
   "analyzed fine, not convertible" — read `result.unsupported`.

2. **Convert** — `coreai-onnx convert model.onnx -o model.aimodel --json`

   ```json
   {
     "schema_version": 1,
     "command": "convert",
     "status": "ok",
     "result": {
       "output_path": "/var/folders/f9/k__xp4rn4_97h3q7p8597k840000gn/T/tmp6ccnaupb/relu.aimodel",
       "total_nodes": 1,
       "optimized": true,
       "validated": true,
       "repairs": [],
       "precision": {
         "passed": true,
         "rtol": null,
         "atol": null,
         "min_psnr": null,
         "compute_unit": null,
         "seed": 0,
         "outputs": [
           {
             "name": "out0",
             "max_abs_error": 0.0,
             "max_rel_error": 0.0,
             "psnr": "inf",
             "passed": true,
             "expected_nonfinite": 0
           }
         ]
       }
     },
     "warnings": [],
     "error": null
   }
   ```

   On success, `result.precision` carries the ONNX Runtime comparison when it
   ran; `warnings` explains skipped steps (`onnxruntime_missing`,
   `platform_no_runtime`).

   Add `--repair` to auto-fix documented Core AI runtime limitations (e.g.
   float16) with known-safe, parity-verified rewrites; applied fixes are listed
   in `result.repairs`.

3. **On failure, branch on `error.code`** — here a model containing the
   unsupported `Det` op (exit code 1):

   ```json
   {
     "schema_version": 1,
     "command": "convert",
     "status": "error",
     "result": null,
     "warnings": [],
     "error": {
       "code": "unsupported_ops",
       "message": "The following ONNX ops have no Core AI lowering:\n  Det (1 node(s), e.g. node_0)\n\nRegister a custom lowering to proceed:\n    @converter.register_onnx_lowering(\"Det\")\n    def lower(values_map, node, loc): ...\nRun `coreai-onnx inspect <model>` for a full coverage report.",
       "details": {
         "missing": {
           "Det": [
             "node_0"
           ]
         }
       },
       "hint": "Register a custom lowering with @converter.register_onnx_lowering, or run `coreai-onnx inspect <model>` for a full coverage report."
     }
   }
   ```

4. **Re-verify an existing asset** — `coreai-onnx verify model.onnx model.aimodel --json`
   re-runs the precision check against ONNX Runtime without converting again
   (requires macOS 27+ with the Core AI runtime).

## Error codes and recovery

| Code | What happened | What to do |
|---|---|---|
| `unsupported_ops` | ops with no lowering; `details.missing` maps op → example nodes | report the list; options: register a custom lowering (see docs), or change the export |
| `invalid_model_file` | not a valid ONNX model | check the path/produce a valid export |
| `io_error` | file unreadable/unwritable | check path and permissions |
| `model_validation_failed` | model does not run on ONNX Runtime | the input model is broken — fix the export, not the converter |
| `conversion_failed` | one lowering failed (`details.node_name`/`op_key` when known, else `details: null`) | report; often a model-specific edge — file an issue with the node |
| `compiler_failed` | Core AI compiler rejected the program | report the MLIR diagnostic in `message` |
| `precision_check_failed` | converted, but outputs exceeded tolerance | the `.aimodel` EXISTS; inspect `result.precision` — a high PSNR (> 60 dB) means benign accumulation noise: re-run with `--min-psnr`; a NaN on the default units that disappears with `--compute-unit cpu_only` means the model is float16-unstable on GPU/ANE (conversion itself is correct); otherwise consider `--rtol/--atol`. See docs on benign causes |
| `precision_check_error` | converted, but the check crashed | the `.aimodel` EXISTS; retry `verify` separately |
| `platform_unsupported` | `verify` needs macOS 27+ | run verify on a Mac; conversion output is unaffected |

Warnings: `onnxruntime_missing` (install the `[verify]` extra to enable
validation), `platform_no_runtime` (precision check skipped off-macOS),
`reference_nonfinite` (the input model itself produces NaN/Inf on the random
probe input; parity at those positions is checked by NaN/Inf mask and
`result.precision.outputs[*].expected_nonfinite` counts them — the conversion
is fine, the model's numerics on random data are not).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | failure (bad model, unsupported ops, conversion/verification failure) |
| 2 | usage error (argparse on stderr, no JSON) or platform error (`platform_unsupported`, with a JSON envelope) |
| 3 | `.aimodel` written but the precision check failed or could not run |

Exit 3 from `convert` and `precision_check_failed`/`precision_check_error`
mean you HAVE a converted model — decide whether the precision report is
acceptable instead of treating it as fatal. (From `verify`, the same two
codes exit 1: nothing was written, the check simply failed.)

## Stability

`schema_version` (currently 1) bumps only on breaking envelope changes; error
codes, warning codes, and exit codes are append-only. Pin nothing else.

## Python API

The same pipeline is available as a library (`coreai_onnx.convert`,
`analyze`, `verify`, `supported_ops`) returning typed objects — see the
[docs](https://devin-lai.github.io/coreai-onnx/). Note `verify` is
`async`: `await` it or wrap it in `asyncio.run(...)`.

## Attribution

coreai-onnx is BSD-3-Clause licensed and free to use, including commercially.
When you build on it — in generated code, reports, or derived tooling — please
cite the source: https://github.com/devin-lai/coreai-onnx
