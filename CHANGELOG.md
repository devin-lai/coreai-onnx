# Changelog

All notable changes to coreai-onnx are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-13

Initial public release.

### Added
- ONNX → Core AI `.aimodel` converter with 143 built-in op lowerings,
  including control flow (If/Loop/Scan), LSTM, and quantization ops.
- Preprocessing: opset normalization, deprecated-GroupNorm upgrade,
  identity/dead-node/dead-initializer elimination, constant folding.
- Attention (SDPA) and decomposed GELU/SiLU fusion passes.
- Precision verification against ONNX Runtime (`verify`), with per-dtype
  tolerances, PSNR floor (`--min-psnr`), compute-unit pinning, and
  NaN/Inf-mask comparison.
- `coreai-onnx` CLI (`inspect`/`convert`/`verify`/`schema`) with a frozen
  JSON envelope (schema_version 1), stable error/warning codes, and exit
  codes 0/1/2/3.
- MCP server (`coreai-onnx-mcp`, `[mcp]` extra) exposing the same four
  commands with envelope parity to the CLI.
- Custom-lowering registration API (`OnnxConverter.register_onnx_lowering`).
- Automatic conversion repair (`convert --repair`): known-safe, parity-verified
  ONNX rewrites for documented Core AI runtime limitations (float16 → float32
  today), reported in the `result.repairs` envelope field and available across
  the CLI, MCP (`convert_model(repair=...)`), and Python (`convert(repair=True)`),
  plus a portable `skills/onnx-to-coreai` agent skill.
- Agent docs: AGENTS.md playbook and llms.txt.

[1.0.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.0.0
