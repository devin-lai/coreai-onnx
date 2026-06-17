# Changelog

All notable changes to coreai-onnx are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [1.1.1] - 2026-06-17

### Fixed
- Match ONNX Runtime's float32 truncation behavior when deriving Resize output
  dimensions, restoring exact parity across the covered resize cases.

## [1.1.0] - 2026-06-15

### Changed
- **The default `convert`/`verify` precision verdict now measures fp32
  conversion fidelity.** The check still runs first on the runtime's fast
  default compute unit, but on a miss it falls back to the deterministic fp32
  `cpu_only` path for the verdict (the fallback runs *only* on a miss, so a
  passing conversion pays no extra cost). Two correct-but-divergent classes that
  previously failed now pass with a warning: float16 GPU/ANE divergence —
  including NaN/overflow and a runtime abort — surfaces as
  `precision_hardware_divergence` when the model is faithful on fp32, and benign
  large-magnitude accumulation noise (high PSNR) surfaces as
  `precision_benign_noise`. `--min-psnr` now sets the benign floor (default
  50 dB). `result.precision.compute_unit` echoes the *effective* unit behind the
  verdict (`null` = default unit, `"cpu_only"` = fp32 fallback, or the pinned
  unit). An explicit `--compute-unit` is honored exactly, with no fallback —
  pin `gpu`/`ane` to make the float16 hardware path the verdict.

### Added
- Warning codes `precision_benign_noise` and `precision_hardware_divergence`
  (append-only contract additions; surfaced by the schema dump, CLI, and MCP).
- `verify(..., isolate_execution=...)` and `OutputReport.elementwise_passed`.

### Fixed
- The post-conversion precision check now executes the converted `.aimodel`
  out-of-process. The native Core AI runtime can `abort()` (not raise) on a
  program it cannot execute on the selected compute unit — e.g. an MPSGraph
  `IOSurface` assertion on the GPU path — which previously killed the whole CLI
  invocation (losing its JSON envelope) or the long-lived MCP server. Such an
  abort is now contained and surfaced as a structured `precision_check_error`
  (convert exit 3 / verify exit 1); the `.aimodel` is still written and the
  conversion is unaffected. The error message and docs point to
  `--compute-unit cpu_only` for recovery rather than a blind retry.

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

[1.1.1]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.1.1
[1.1.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.1.0
[1.0.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.0.0
