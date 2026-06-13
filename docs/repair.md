# Automatic repair

`coreai-onnx convert --repair` (and `convert_model(repair=True)` over MCP, or
`coreai_onnx.convert(model, repair=True)` in Python) applies **known-safe**
rewrites for documented Core AI runtime limitations, then re-verifies parity
against ONNX Runtime before trusting the result.

## What it guarantees

- **Only documented, known-safe rewrites.** Each repair is a
  semantics-preserving ONNX → ONNX transform tied to a confirmed Core AI runtime
  limitation.
- **No silently-wrong models.** After repairing, the precision check compares
  the `.aimodel` against ONNX Runtime. A rewrite that would change results
  beyond the original precision is rejected, not shipped.
- **Opt-in and additive.** Without `--repair`, conversion behaves exactly as
  before. With it, `result.repairs` lists every fix applied.

## What it fixes today

| Repair | Trigger | Rewrite |
|--------|---------|---------|
| `promote_float16_to_float32` | float16 graph inputs | Promote all float16 tensors (inputs, initializers, outputs) to float32. The Core AI runtime cannot load most float16 programs (`Program load failure 0x10004`); float32 is a strict superset, so results match within float16 precision. Inputs and outputs become float32. |

More strategies are added as Core AI runtime limitations are confirmed — each
ships with its own parity test. As CoreAI fixes a limitation upstream, the
corresponding repair is retired. The repair registry lives in
`coreai_onnx._repair`.

## Example

```bash
coreai-onnx convert model.onnx -o model.aimodel --repair --json
```

```json
{
  "status": "ok",
  "result": {
    "output_path": "model.aimodel",
    "repairs": [
      {
        "name": "promote_float16_to_float32",
        "summary": "promote float16 tensors to float32 ...",
        "details": {"inputs": ["x"]}
      }
    ],
    "precision": {"passed": true}
  }
}
```

## Using it from an agent

The `skills/onnx-to-coreai` skill packages this workflow as portable
instructions any AI agent can follow — it simply drives the CLI shown above and
branches on the JSON envelope. See the
[agent playbook](https://github.com/devin-lai/coreai-onnx/blob/main/AGENTS.md).
