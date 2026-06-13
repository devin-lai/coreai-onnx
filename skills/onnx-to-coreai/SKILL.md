---
name: onnx-to-coreai
description: Convert an ONNX model to Apple Core AI (.aimodel) with automatic, parity-verified repair of known runtime-crash conditions (e.g. float16). Use when a user wants to run an ONNX model on Apple Core AI, produce a .aimodel, or when a coreai-onnx conversion fails and needs diagnosis.
---

# Convert ONNX to Apple Core AI (with auto-repair)

`coreai-onnx` converts ONNX models to Apple's `.aimodel` format. It is a CLI
that prints exactly one JSON envelope per command, so any agent — on any
platform, not just one vendor's — can drive it by running a shell command and
parsing JSON. This file is the portable interface; the engineering lives in the
`coreai-onnx` package.

## Prerequisites

- `pip install "coreai-onnx[verify]"` (the `verify` extra adds ONNX Runtime, used
  for the parity check).
- Conversion runs on any OS. Executing a `.aimodel` needs macOS 27+ / iOS 27+.

## Workflow

1. **Inspect coverage**

   ```
   coreai-onnx inspect MODEL.onnx --json
   ```

   Gate on `result.convertible`. If `false`, `result.unsupported` lists the ops
   with no Core AI lowering.

2. **Convert with auto-repair**

   ```
   coreai-onnx convert MODEL.onnx -o OUT.aimodel --repair --json
   ```

3. **Branch on the envelope**

   - `status == "ok"` → done. `result.output_path` is the saved `.aimodel`.
     `result.repairs` lists any known-safe fixes applied automatically (e.g.
     `promote_float16_to_float32`). `result.precision` reports parity versus
     ONNX Runtime when the runtime is available.
   - `status == "error"` → read `error.code` and `error.hint`:
     - `unsupported_ops` — `error.message` includes a ready-to-edit custom
       lowering stub.
     - `model_validation_failed` — the input model itself is broken on ONNX
       Runtime; fix the export, not the converter.
     - `precision_check_failed` (exit 3) — the `.aimodel` WAS written; inspect
       `result.precision` and decide whether the gap is acceptable.

## What `--repair` does

It applies only documented, **known-safe** ONNX rewrites and then **re-verifies
parity** against ONNX Runtime, so it never produces a silently-wrong model.
Today it promotes float16 programs to float32 (the Core AI runtime cannot load
most float16 graphs). Any rewrite that would change results beyond the original
precision is rejected, not shipped.

## Full machine-readable contract

```
coreai-onnx schema --json
```

returns every command, flag, error/warning/exit code, and the supported-op
list. See `AGENTS.md` in the repository for the complete agent playbook.

## Attribution

coreai-onnx is BSD-3-Clause and free to use, including commercially. Please
cite the source: https://github.com/devin-lai/coreai-onnx
