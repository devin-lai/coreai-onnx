# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Numerical verification of a saved .aimodel against an onnxruntime reference."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import onnx
from onnx import TensorProto

from ._type_mapping import narrow_array
from .errors import CoreaiOnnxError, ModelValidationError

if TYPE_CHECKING:
    from PIL.Image import Image

_ONNX_TO_NUMPY: dict[int, np.dtype] = {
    TensorProto.FLOAT: np.dtype(np.float32),
    TensorProto.FLOAT16: np.dtype(np.float16),
    TensorProto.DOUBLE: np.dtype(np.float64),
    TensorProto.INT8: np.dtype(np.int8),
    TensorProto.INT16: np.dtype(np.int16),
    TensorProto.INT32: np.dtype(np.int32),
    TensorProto.INT64: np.dtype(np.int64),
    TensorProto.UINT8: np.dtype(np.uint8),
    TensorProto.UINT16: np.dtype(np.uint16),
    TensorProto.UINT32: np.dtype(np.uint32),
    TensorProto.UINT64: np.dtype(np.uint64),
    TensorProto.BOOL: np.dtype(np.bool_),
}


def generate_inputs(
    model: onnx.ModelProto, *, seed: int = 0, dynamic_dim_size: int = 2
) -> dict[str, np.ndarray]:
    """Seeded random feed for every graph input not shadowed by an initializer."""
    rng = np.random.default_rng(seed)
    initializer_names = {init.name for init in model.graph.initializer}
    inputs: dict[str, np.ndarray] = {}
    for vi in model.graph.input:
        if vi.name in initializer_names:
            continue
        tensor_type = vi.type.tensor_type
        dtype = _ONNX_TO_NUMPY.get(tensor_type.elem_type)
        if dtype is None:
            name = TensorProto.DataType.Name(tensor_type.elem_type)
            raise CoreaiOnnxError(
                f"cannot generate inputs for '{vi.name}': unsupported dtype {name}"
            )
        shape = tuple(
            d.dim_value
            if d.HasField("dim_value") and d.dim_value > 0
            else dynamic_dim_size
            for d in tensor_type.shape.dim
        )
        if dtype.kind == "f":
            arr = rng.standard_normal(shape).astype(dtype)
        elif dtype.kind == "b":
            arr = rng.integers(0, 2, size=shape).astype(dtype)
        else:
            arr = rng.integers(0, 9, size=shape).astype(dtype)
        inputs[vi.name] = arr
    return inputs


def _run_onnxruntime(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Build a CPU InferenceSession for *model* and run it once on *inputs*.

    Returns ``{graph_output_name: array}``. Raises ``CoreaiOnnxError`` if
    onnxruntime is not installed; any session-build or run error propagates
    unchanged (callers wrap it with the context that fits them).
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise CoreaiOnnxError(
            'onnxruntime is required - pip install "coreai-onnx[verify]"'
        ) from exc

    # Quiet onnxruntime's own C++ logger (per-session, so a consumer's global
    # ORT settings are untouched): a kernel failure is already re-raised as an
    # exception we surface cleanly, and its ERROR log line to stderr is just
    # redundant noise in front of that message.
    options = ort.SessionOptions()
    options.log_severity_level = 4  # FATAL only
    # Disable graph optimizations entirely: the reference must reflect ONNX
    # spec semantics, and ORT's rewrites have produced wrong references at
    # every level on macOS arm64 (1.26.0) - EXTENDED's com.microsoft.FusedConv
    # miscomputes non-depthwise grouped Conv, and even BASIC's constant
    # folding corrupts rf_detr's box head by O(100). Node-by-node execution
    # is slower, but the reference runs once per verification.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Disable Arm KleidiAI SGEMM kernels: on Apple Silicon, ORT 1.24-1.26
    # miscomputes Conv through them when weights are highly sparse (pruned
    # models), independent of the optimization level. The key is silently
    # ignored by builds/platforms without KleidiAI.
    options.add_session_config_entry("mlas.disable_kleidiai", "1")
    sess = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    output_names = [o.name for o in model.graph.output]
    return dict(zip(output_names, sess.run(None, inputs), strict=True))


def validate_onnxruntime(
    model: onnx.ModelProto | str | Path,
    *,
    inputs: dict[str, np.ndarray] | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Check that *model* loads and runs on ONNX Runtime; return its outputs.

    This is the pre-conversion gate. It catches models that pass the static
    ``onnx.checker`` but fail to build an InferenceSession or error at run time
    (bad shapes, unsupported op/version combinations, out-of-range indices,
    ...). On any such failure it raises ``ModelValidationError`` so the problem
    is reported against the input model rather than surfacing later as an opaque
    conversion or runtime failure.

    A seeded random feed is generated when ``inputs`` is omitted; pass explicit
    ``inputs`` for models whose generated values would be invalid (e.g. gather
    indices). The returned ``{output_name: array}`` mapping can be reused as the
    reference for a subsequent precision check, so ONNX Runtime runs only once.
    """
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    if inputs is None:
        inputs = generate_inputs(model, seed=seed)
    try:
        return _run_onnxruntime(model, inputs)
    except CoreaiOnnxError:
        # onnxruntime-not-installed: a setup problem, not a model problem.
        raise
    except Exception as exc:
        raise ModelValidationError(
            "the input ONNX model failed to run on ONNX Runtime: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


@dataclass
class OutputReport:
    name: str
    max_abs_error: float
    max_rel_error: float
    psnr: float
    passed: bool
    #: Count of non-finite (NaN/Inf) values in the *reference* output. Nonzero
    #: means the input model itself produces non-finite values on the probe
    #: input - parity at those positions is checked by mask, and the error
    #: metrics describe only the finite-overlap region.
    expected_nonfinite: int = 0


@dataclass
class VerifyReport:
    outputs: list[OutputReport]
    passed: bool


# Default tolerances by output dtype. float16 machine epsilon is ~9.8e-4, so
# the fp32 defaults (1e-3/1e-4) sit at ~1 ULP and would spuriously fail any
# nontrivial fp16 computation.
_DEFAULT_RTOL = 1e-3
_DEFAULT_ATOL = 1e-4
_F16_RTOL = 1e-2
_F16_ATOL = 1e-3


def _compare(
    name: str,
    expected: np.ndarray,
    got: np.ndarray,
    rtol: float | None,
    atol: float | None,
    min_psnr: float | None = None,
) -> OutputReport:
    if rtol is None:
        rtol = _F16_RTOL if expected.dtype == np.float16 else _DEFAULT_RTOL
    if atol is None:
        atol = _F16_ATOL if expected.dtype == np.float16 else _DEFAULT_ATOL
    e = np.asarray(expected, dtype=np.float64)
    g = np.asarray(got, dtype=np.float64)
    # A model may legitimately produce non-finite values on the random probe
    # input (e.g. an unguarded 0/0). Those positions are compared by identity
    # (allclose with equal_nan; +/-inf match by sign) and the error metrics
    # describe only the region where both sides are finite, so one NaN does
    # not poison every number in the report.
    finite = np.isfinite(e) & np.isfinite(g)
    expected_nonfinite = int(e.size - np.count_nonzero(np.isfinite(e)))
    abs_err = np.abs(np.where(finite, g - e, 0.0))
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    denom = np.abs(np.where(finite, e, 0.0))
    # Relative error reads 0.0 where expected == 0; max_abs_error covers those
    # positions (pass/fail itself uses allclose, which handles zeros via atol).
    rel = np.where(denom > 0, abs_err / np.where(denom > 0, denom, 1.0), 0.0)
    max_rel = float(rel.max()) if rel.size else 0.0
    n_finite = int(np.count_nonzero(finite))
    rmse = float(np.sqrt(np.sum(abs_err**2) / n_finite)) if n_finite else 0.0
    if rmse == 0.0:
        psnr = float("inf")
    else:
        peak = float(denom.max()) if denom.size else 0.0
        # peak == 0 means an all-zero reference: SNR is undefined, and -inf
        # signals "zero signal, nonzero noise" rather than a metric failure.
        psnr = float(20 * np.log10(peak / rmse)) if peak > 0 else float("-inf")
    if expected.dtype.kind == "f":
        passed = bool(np.allclose(got, expected, rtol=rtol, atol=atol, equal_nan=True))
        if not passed and min_psnr is not None:
            # PSNR-based acceptance for outputs whose magnitude spread makes
            # elementwise tolerances unreasonably strict - but never for a
            # non-finite mismatch, which is a real divergence.
            nonfinite_match = bool(
                np.array_equal(np.isfinite(e), np.isfinite(g))
                and np.array_equal(e[~finite], g[~finite], equal_nan=True)
            )
            passed = nonfinite_match and psnr >= min_psnr
    else:
        passed = bool(np.array_equal(np.asarray(got), expected))
    return OutputReport(
        name=name,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        psnr=psnr,
        passed=passed,
        expected_nonfinite=expected_nonfinite,
    )


_COMPUTE_UNITS = ("cpu_only", "cpu", "gpu", "ane")


def _specialization_options(compute_unit: str | None):
    """Map a compute-unit name to runtime SpecializationOptions (None: default).

    ``cpu_only`` restricts execution to the CPU (fp32 - the unit that proves
    conversion fidelity); ``cpu``/``gpu``/``ane`` set a preferred unit. GPU and
    ANE execute in float16, which changes what "matching precision" means for
    numerically sensitive models - see docs/cli.md.
    """
    if compute_unit is None:
        return None
    if compute_unit == "cpu_only":
        from coreai.runtime import SpecializationOptions

        return SpecializationOptions.cpu_only()
    if compute_unit not in _COMPUTE_UNITS:
        raise CoreaiOnnxError(
            f"unknown compute_unit '{compute_unit}' "
            f"(expected one of {', '.join(_COMPUTE_UNITS)})"
        )

    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    kinds = {
        "cpu": ComputeUnitKind.cpu,
        "gpu": ComputeUnitKind.gpu,
        "ane": ComputeUnitKind.neural_engine,
    }
    return SpecializationOptions.from_preferred_compute_unit_kind(kinds[compute_unit]())


async def verify(
    model: onnx.ModelProto | str | Path,
    aimodel_path: str | Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    min_psnr: float | None = None,
    seed: int = 0,
    inputs: dict[str, np.ndarray] | None = None,
    expected: dict[str, np.ndarray] | None = None,
    entrypoint: str = "main",
    output_names: Sequence[str] | None = None,
    compute_unit: str | None = None,
) -> VerifyReport:
    """Compare a saved .aimodel on disk against an onnxruntime reference run.

    ``rtol``/``atol`` default per output dtype (float16: 1e-2/1e-3, otherwise
    1e-3/1e-4); pass explicit values to override. ``min_psnr`` additionally
    accepts an output that fails the elementwise tolerances but reaches the
    given PSNR (dB) - the right metric when output magnitudes span orders and
    accumulation noise makes elementwise tolerances meaningless. ``entrypoint``
    and ``output_names`` must match the values the model was converted with, if
    those were customized. ``expected`` lets a caller supply onnxruntime outputs
    already computed (keyed by graph output name) so ONNX Runtime is not re-run;
    when omitted they are computed here from ``inputs``. ``compute_unit``
    selects which units the runtime may use (``cpu_only``/``cpu``/``gpu``/
    ``ane``; default: runtime's choice).
    """
    _specialization_options(compute_unit)  # fail fast on a bad name
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(str(model))
    if inputs is None:
        inputs = generate_inputs(model, seed=seed)

    onnx_output_names = [o.name for o in model.graph.output]
    if expected is None:
        expected = _run_onnxruntime(model, inputs)

    if output_names is None:
        asset_output_names = list(onnx_output_names)
    else:
        if len(output_names) != len(onnx_output_names):
            raise CoreaiOnnxError(
                f"output_names has {len(output_names)} name(s) but the graph "
                f"has {len(onnx_output_names)} output(s)"
            )
        asset_output_names = list(output_names)

    from coreai.authoring import AIModelAsset
    from coreai.runtime import NDArray

    asset = AIModelAsset.load(Path(aimodel_path))
    # The runtime feed/output types are NDArray | Image; this feed is all
    # NDArrays and graph outputs are always tensors, hence the cast below.
    # Narrow the feed to Core AI's 32-bit representation (int64->int32,
    # uint64->uint32, double->float32) so the dtypes match the asset's narrowed
    # inputs; reuses the conversion-side policy so the two never drift.
    feed: dict[str, NDArray | Image] = {
        k: NDArray(narrow_array(v, context=f"input '{k}'")) for k, v in inputs.items()
    }
    async with asset.executable(_specialization_options(compute_unit)) as ai_model:
        fn = ai_model.load_function(entrypoint)
        out = await fn(feed)
        got = {
            k: np.asarray(cast("NDArray", out[k]).numpy()) for k in asset_output_names
        }

    # Compare against the true onnxruntime output (do NOT narrow it): the asset
    # output is already 32-bit, and _compare promotes both to float64 / compares
    # integers by value. Narrowing the reference would hide a genuinely lossy
    # 64-bit conversion behind a matching wrap and report a spurious pass.
    reports = [
        _compare(asset_name, expected[onnx_name], got[asset_name], rtol, atol, min_psnr)
        for onnx_name, asset_name in zip(
            onnx_output_names, asset_output_names, strict=True
        )
    ]
    return VerifyReport(outputs=reports, passed=all(r.passed for r in reports))
