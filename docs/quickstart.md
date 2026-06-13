# Quickstart

## Installation

```bash
pip install coreai-onnx
```

To enable numerical verification against ONNX Runtime (requires `onnxruntime`):

```bash
pip install "coreai-onnx[verify]"
```

## Convert a model

```python
from pathlib import Path

import coreai_onnx

program = coreai_onnx.convert("model.onnx", input_names=["x"], output_names=["y"])
program.optimize()
program.save_asset(Path("model.aimodel"))
```

That is the entire API for the common case. `convert()` accepts either a path string or
an `onnx.ModelProto` loaded in memory.

## CLI examples

**Inspect coverage before converting:**

```bash
coreai-onnx inspect model.onnx
```

**Convert to `.aimodel`:**

```bash
coreai-onnx convert model.onnx -o model.aimodel
```

**Verify numerical parity against ONNX Runtime:**

```bash
coreai-onnx verify model.onnx model.aimodel
```

## Attention fusion

Raw scaled-dot-product attention chains (`MatMul → scale → Softmax → MatMul`,
as exported by PyTorch, YOLO, BERT, etc.) crash the on-device GPU compiler —
the symptom is Xcode's Performance report aborting with
`MPSGraphExecutable: "MLIR pass manager failed"`. The converter therefore
fuses these chains automatically into the same `scaled_dot_product_attention`
composite op that `coreai-torch` emits, which the OS compilers replace with
their fused attention kernel. The rewrite is numerically equivalent; chains
that do not match the conservative pattern (dynamic shapes, masked softmax on
a non-final axis, multi-consumer intermediates, …) are left untouched and
convert exactly as before.

You can also instantiate the composite directly from a custom ONNX graph with
a `coreai::ScaledDotProductAttention` node — `query [.., L, E]`,
`key [.., S, E]`, `value [.., S, E]` inputs (rank 3 or 4), an optional float
additive mask, and a float `scale` attribute (default `E**-0.5`).

## Runtime note

**Conversion** (producing `.aimodel` files) works on any platform: macOS, Linux,
Windows. The converter does not invoke the Core AI runtime.

**Execution** of `.aimodel` files requires **macOS 27+** or **iOS 27+** with the Core
AI framework present. Parity tests and the `verify` command require the runtime and
will be skipped automatically on unsupported platforms.

## Next steps

- [Supported ops](coverage.md) — 143 built-in op lowerings supported today
- [CLI reference](cli.md) — all subcommands and exit codes
- [Custom lowerings](custom-lowerings.md) — extend coverage for your own ops
