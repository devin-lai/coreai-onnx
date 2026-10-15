# Changelog

All notable changes to coreai-onnx are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Verification rejects mismatched output shapes, preserves zero-length input
  dimensions and float64 NaN/Inf values, and computes PSNR without overflow or
  underflow from squaring errors. Precision warnings now appear in human CLI
  output and report the requested PSNR threshold correctly.
- Graph cleanup and fusion preserve custom-domain operators. Identity removal
  respects shadowed source names in subgraphs, and GELU fusion preserves ranks
  introduced by broadcasting constants.
- Dynamic `Flatten`, scalar `GatherND`, infinite-input `ReduceLogSumExp`, and
  scalar attention masks now convert and execute correctly.
- `MeanVarianceNormalization` includes the ONNX-specified `1e-9` epsilon after
  the standard deviation. Some ONNX Runtime versions omit this epsilon; checks
  on constant or very small inputs may therefore report a reference discrepancy.
- Constant folding checks expanding tensors before allocation. Large constant
  `Range` and `ConstantOfShape` operations remain compact during lowering.
- Failed reconversions preserve existing `.aimodel` bundles. Replacements are
  staged before installation, with rollback and backup retention on I/O errors.
- Runtime checks enforce macOS 27+, and `verify` reports malformed model files
  as `invalid_model_file` rather than a precision-check error.
- The MCP extra requires SDK 1.10.0+, which introduced structured tool outputs,
  and tool annotations remain compatible with that minimum version.

## [1.2.0] - 2026-10-11

Compatibility updates for coreai-core 1.0.0b3, with operator fixes and broader
dynamic-shape support.

### Changed
- **`convert()` and `OnnxConverter.to_coreai()` now return an optimized
  program.** coreai-core 1.0.0b3 removed `AIProgram.optimize()`, causing default
  CLI conversions to fail with `compiler_failed`. Optimization now runs inside
  the converter. Remove `program.optimize()` from existing code, or pass
  `optimize=False` to inspect the IR before optimization. `--no-optimize` and
  the envelope's `optimized` field are unchanged.
- Requires `coreai-core>=1.0.0b2` (was `>=1.0.0b1`): 1.0.0b1 still converts,
  but its runtime aborts when executing an asset on the current macOS 27 build,
  so it can no longer be validated.
- The `[mcp]` extra now requires `mcp<2`. MCP SDK 2.x renamed `FastMCP`, which
  prevented the server from starting. The import error now identifies the
  installed SDK version and explains how to install a supported version.
- Python 3.14 is supported (`requires-python` is now `>=3.11,<3.15`).
- `Resize` now follows ONNX Runtime's float32 coordinate calculations and
  tolerance for `.5` ties. Dimensions with a scale of 1 are left unchanged,
  and the input is copied when the output shape matches it.

### Added
- `ArgMax`/`ArgMin` with `select_last_index=1`.
- `ReduceMean` on integer tensors (the truncated integer mean).
- `Pad` with negative constant pads (cropping) and with non-negative pad
  amounts computed at runtime.
- `Resize` with a dynamic batch or channel dimension (H and W stay static).
- `Concat` of inputs that disagree on which dimensions are static.

### Fixed
- `Floor`, `Ceil`, and `Round` returned their input unchanged in every
  optimized program, and on `cpu_only` even unoptimized: Core AI fuses the
  float→int32→float cast pair the lowering was built on into a no-op. They now
  lower as coreai-torch does (`floor_divide(x, 1)`, native `round`), which also
  makes them correct on `cpu`/`cpu_only`; float16 inputs are computed in
  float32.
- `Cast` float→int→float (`x.long().float()`) lost its truncation to the same
  fusion, and float→bool→float lost its binarization on `cpu_only`. Casts to
  bool are now a `!= 0` comparison and the int→float half of a round trip is an
  explicit truncation.
- Integer `Div` next to constant `Mul`/`Sub` was simplified as real-number
  division — `(x / 48) * 48` became `x`, `x - (x / 48) * 48` became `0`,
  `(x * 3) / 2` became `x` — corrupting the shape arithmetic of dynamic models.
- `Gather` with a scalar index on a 1-D tensor (`tensor.shape[i]`) aborted the
  process on the default and GPU compute units.
- `ArgMin` over a dynamically sized axis with `keepdims=0` failed to compile.
- The precision check failed with `TypeError: Unable to create NDArray` for
  models with a rank-0 int64 or double input.
- Large `Trilu` masks were folded back into a dense constant by the optimizer
  (1 MiB for a 1024 context, against ~10 KiB unfolded); folds that expand a
  tensor past 65,536 elements are now left to run at inference.
- Importing coreai-onnx no longer depends on an incidental import order to
  avoid a circular import inside coreai-core 1.0.0b3.

## [1.1.2] - 2026-06-17

### Changed
- Detect float16 use inside graphs before applying repair rewrites.
- Add provenance metadata when saving bundles.
- Pin NumPy below the next major release pending validation.

### Fixed
- Reject invalid split partitions and mismatched lowering result counts.

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

[1.2.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.2.0
[1.1.2]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.1.2
[1.1.1]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.1.1
[1.1.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.1.0
[1.0.0]: https://github.com/devin-lai/coreai-onnx/releases/tag/v1.0.0
