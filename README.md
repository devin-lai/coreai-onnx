# coreai-onnx

[![Lightweight CI](https://github.com/devin-lai/coreai-onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/devin-lai/coreai-onnx/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/coreai-onnx)](https://pypi.org/project/coreai-onnx/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The missing ONNX path into Apple's Core AI.**

coreai-onnx converts ONNX models directly to the `.aimodel` format consumed by
Apple's Core AI framework — the AI-first successor to Core ML on iOS 27 and
macOS 27 — with no manual model surgery and no custom export scripts.

## Quickstart

```python
from pathlib import Path

import coreai_onnx

program = coreai_onnx.convert("model.onnx", input_names=["x"], output_names=["y"])
program.optimize()
program.save_asset(Path("model.aimodel"))
```

Or from the command line:

```bash
coreai-onnx inspect model.onnx                     # check coverage before converting
coreai-onnx convert model.onnx -o model.aimodel    # validates on ONNX Runtime, converts, then reports precision
coreai-onnx verify  model.onnx model.aimodel       # re-check precision of an existing .aimodel
coreai-onnx schema  --json                         # machine-readable capability dump
```

`convert` runs an end-to-end pipeline: it first confirms the model loads and runs
on ONNX Runtime (and stops with a clear error if it does not), converts and
optimises it, then prints a precision comparison of the saved `.aimodel` against
ONNX Runtime. The ONNX Runtime steps are skipped, with a note, when `onnxruntime`
is not installed or you are not on macOS.

A real session on torchvision's MobileNetV3-Small (opset 18):

```text
$ coreai-onnx inspect mobilenet_v3_small.onnx
┃ Op                ┃ Count ┃ Status    ┃
│ Add               │     6 │ supported │
│ Conv              │    52 │ supported │
│ Gemm              │     2 │ supported │
│ GlobalAveragePool │    10 │ supported │
│ HardSwish         │    19 │ supported │
│ ...               │       │           │
Total nodes: 159  |  convertible: yes

$ coreai-onnx convert mobilenet_v3_small.onnx -o mobilenet_v3_small.aimodel
Converted successfully
  Nodes: 159
  Output: mobilenet_v3_small.aimodel
       Precision: ONNX Runtime vs .aimodel
┃ Output ┃ Max abs error ┃ Max rel error ┃   PSNR ┃ Pass ┃
│ 400    │    7.7716e-16 │    9.1514e-03 │ 136.28 │ PASS │
Precision check passed
```

## For AI agents

coreai-onnx is built to be driven by agents: every command takes `--json` and
returns one stable envelope with documented error and exit codes.

- **[AGENTS.md](https://github.com/devin-lai/coreai-onnx/blob/main/AGENTS.md)** — the agent playbook: workflow, error recovery, exit codes.
- **MCP server** — `pip install "coreai-onnx[mcp]"`, then register `coreai-onnx-mcp` (stdio); tools return the same JSON envelope ([docs](https://devin-lai.github.io/coreai-onnx/mcp.html)).
- **`coreai-onnx schema --json`** — the full machine-readable capability contract.
- **[llms.txt](https://devin-lai.github.io/coreai-onnx/llms.txt)** — the llms.txt index, served from the docs site.

## Install

```bash
pip install coreai-onnx                    # conversion only
pip install "coreai-onnx[verify]"          # + numerical verification via onnxruntime
```

## Features

- **143 built-in op lowerings** supported out of the box — elementwise,
  reductions, matrix math, convolutions, normalisation, quantization, control
  flow, shape manipulation, and more.
- **Parity-verified** — every built-in lowering is tested against ONNX Runtime
  on randomised inputs.
- **Validated, precision-checked conversion** — `convert` confirms the input
  model actually runs on ONNX Runtime *before* converting (failing fast with a
  clear error on a broken model), then reports the precision of the saved
  `.aimodel` against ONNX Runtime so you see the numerical gap up front.
- **Automatic attention fusion** — scaled-dot-product attention chains are
  rewritten into the Core AI `scaled_dot_product_attention` composite, so the
  on-device GPU compiler runs its fused kernel instead of crashing on the raw
  `MatMul → Softmax → MatMul` pattern (the cause of Xcode Performance-report
  failures on transformer/YOLO models).
- **Automatic activation fusion** — decomposed GELU and SiLU/Swish chains,
  including chains inside ONNX control-flow branches, are collapsed to fused
  Core AI primitives for fewer full-tensor kernels and better runtime quality.
- **Inspect before you convert** — `coreai-onnx inspect` shows op coverage and
  a node histogram so you know what to expect before running the full conversion.
- **Custom lowering escape hatch** — register a Python function to lower any
  op your model needs, including custom-domain ops:

  ```python
  @converter.register_onnx_lowering("mycompany::MyOp")
  def lower_my_op(values_map, node, loc):
      ...
  ```

- **Conversion runs anywhere** — macOS, Linux, or Windows; no Apple hardware
  needed for the conversion step itself.

## Runtime requirements

| Task | Platform |
|------|----------|
| Convert ONNX → `.aimodel` | Any (macOS, Linux, Windows) |
| Execute `.aimodel` | macOS 27+ / iOS 27+ (Core AI framework) |
| Verify parity (`verify` subcommand) | macOS 27+ + `onnxruntime` |

## CI scope

Public GitHub CI is intentionally lightweight. GitHub-hosted runners do not
currently provide the macOS 27 / iOS 27 SDK / Xcode 27 beta environment needed
to validate Apple Core AI compiler/runtime behavior, so the CI badge covers only
cross-platform safety checks:

```bash
ruff check .
python -m compileall src tests
pytest -m "not apple and not integration"
python -m build
twine check dist/*
```

Hosted CI does not run real Core AI conversion, GPU compilation, `.aimodel`
generation, or Apple SDK integration tests. Full Core AI conversion and runtime
validation must be run locally on macOS 27 + Xcode 27 beta.

## Ecosystem

| Package | Role |
|---------|------|
| **coreai-onnx** | Convert ONNX models to `.aimodel` (this package) |
| coreai-torch | Export PyTorch models directly to Core AI via `torch.export` |
| coreai-opt | Post-conversion optimisation passes (quantisation, pruning) |
| coreai-models | Pre-converted `.aimodel` hub for common model families |

## Documentation

Full documentation including the [supported ops table](https://devin-lai.github.io/coreai-onnx/coverage.html)
and [custom lowering guide](https://devin-lai.github.io/coreai-onnx/custom-lowerings.html)
is published at **https://devin-lai.github.io/coreai-onnx/**.

## Known limitations

The Core AI runtime has a few confirmed quirks. See
[CONTRIBUTING.md — Known runtime quirks](CONTRIBUTING.md#known-runtime-quirks) for
the current list, which includes:

- Float16 graph inputs trigger a runtime crash (cast at the boundary as a workaround).
- Multi-output `If` branches hang at runtime; single-output `If` works.

## Versioning and CoreAI compatibility

coreai-onnx depends on `coreai-core>=1.0.0b1,<2`. Public GitHub CI validates the
linting, packaging, bytecode-compilation, and pure-Python test surface only.
Release validation against the supported Core AI environment is performed
locally on macOS 27 + Xcode 27 beta. The CLI JSON envelope
(`schema_version: 1`), error/warning codes, and exit codes are a frozen,
append-only contract — see [docs/cli.md](docs/cli.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the op-lowering workflow,
and the PR checklist.

## Attribution

coreai-onnx is free to use — including in commercial and closed-source products
— under the [BSD-3-Clause license](LICENSE). In return, please give credit:
clearly attribute the project and cite its source wherever you build on it — in
applications, research, derived tools, and automated or agent workflows alike:

> coreai-onnx — https://github.com/devin-lai/coreai-onnx

And if coreai-onnx is useful to you, please ⭐ **[star the
repository](https://github.com/devin-lai/coreai-onnx)** — real stars from real
users are what help other people discover the project.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
