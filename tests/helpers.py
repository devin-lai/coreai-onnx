# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test harness: build single-op ONNX models, run reference + Core AI, compare."""

import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx._type_mapping import narrow_array

requires_coreai_runtime = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="executing .aimodel requires macOS with the Core AI runtime",
)

_TEST_COMPUTE_UNIT = os.environ.get("COREAI_ONNX_TEST_COMPUTE_UNIT")
_COREAI_RUNTIME_LOCK_PATH = Path(tempfile.gettempdir()) / "coreai_onnx_runtime.lock"


@contextmanager
def _coreai_runtime_lock():
    """Serialize Core AI runtime loads across xdist workers.

    The runtime maintains process-external specialization/cache state. Loading
    several freshly saved .aimodel assets concurrently can fail with transient
    "file does not exist" or "file could not be saved" errors unrelated to the
    converter under test.
    """
    if platform.system() != "Darwin" or os.environ.get(
        "COREAI_ONNX_DISABLE_RUNTIME_LOCK"
    ):
        yield
        return

    import fcntl

    with _COREAI_RUNTIME_LOCK_PATH.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def skip_on_compute_unit(*units: str, reason: str) -> pytest.MarkDecorator:
    """Skip when the suite runs on one of the given units (documented upstream
    per-compute-unit runtime bugs; the emitted IR is identical on all units)."""
    return pytest.mark.skipif(_TEST_COMPUTE_UNIT in units, reason=reason)


_NP_TO_ONNX = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.float16): TensorProto.FLOAT16,
    np.dtype(np.int32): TensorProto.INT32,
    np.dtype(np.int64): TensorProto.INT64,
    np.dtype(np.uint8): TensorProto.UINT8,
    np.dtype(np.int8): TensorProto.INT8,
    np.dtype(np.bool_): TensorProto.BOOL,
}


def single_op_model(
    op_type: str,
    inputs: dict[str, np.ndarray],
    n_outputs: int = 1,
    *,
    attrs: dict[str, Any] | None = None,
    initializers: dict[str, np.ndarray] | None = None,
    opset: int = 22,
) -> onnx.ModelProto:
    """A one-node model. Output dtypes/shapes are filled by shape inference."""
    initializers = initializers or {}
    input_vis = [
        helper.make_tensor_value_info(n, _NP_TO_ONNX[a.dtype], a.shape)
        for n, a in inputs.items()
    ]
    output_names = [f"out{i}" for i in range(n_outputs)]
    node = helper.make_node(
        op_type,
        inputs=list(inputs) + list(initializers),
        outputs=output_names,
        **(attrs or {}),
    )
    graph = helper.make_graph(
        [node],
        f"test_{op_type}",
        input_vis,
        [
            helper.make_tensor_value_info(n, TensorProto.UNDEFINED, None)
            for n in output_names
        ],
        initializer=[
            onnx.numpy_helper.from_array(a, name=n) for n, a in initializers.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def run_onnxruntime(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return sess.run(None, inputs)


def _specialization_options():
    """Optional compute-unit override for the whole parity suite.

    COREAI_ONNX_TEST_COMPUTE_UNIT=cpu|gpu|ane|cpu_only runs every .aimodel
    execution against that compute unit (GPU/ANE-specific miscompilations
    have been observed that the default unit hides). Unset: runtime default.
    """
    kind = os.environ.get("COREAI_ONNX_TEST_COMPUTE_UNIT")
    if not kind:
        return None
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if kind == "cpu_only":
        return SpecializationOptions.cpu_only()
    factory = {
        "cpu": ComputeUnitKind.cpu,
        "gpu": ComputeUnitKind.gpu,
        "ane": ComputeUnitKind.neural_engine,
    }[kind]
    return SpecializationOptions.from_preferred_compute_unit_kind(factory())


async def run_aimodel(
    model: onnx.ModelProto, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    from coreai.runtime import NDArray

    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    program = converter.to_coreai()
    output_names = [o.name for o in model.graph.output]
    with tempfile.TemporaryDirectory() as td, _coreai_runtime_lock():
        asset = program.save_asset(Path(td) / "m.aimodel")
        async with asset.executable(_specialization_options()) as ai_model:
            fn = ai_model.load_function("main")
            feed = {
                k: NDArray(narrow_array(v, context=f"input '{k}'"))
                for k, v in inputs.items()
            }
            out = await fn(feed)
            results = [np.asarray(out[k].numpy()) for k in output_names]
    return results


def relu_model_file(tmp_path) -> str:
    """A trivially-convertible single-op model saved to disk (CLI/MCP tests)."""
    model = single_op_model("Relu", {"x": np.zeros((4,), dtype=np.float32)})
    path = tmp_path / "relu.onnx"
    onnx.save(model, str(path))
    return str(path)


def det_model_file(tmp_path) -> str:
    """A model whose op (Det) has no lowering — the unsupported-ops probe."""
    model = single_op_model("Det", {"X": np.zeros((3, 3), dtype=np.float32)})
    path = tmp_path / "det.onnx"
    onnx.save(model, str(path))
    return str(path)


ENVELOPE_KEYS = {"schema_version", "command", "status", "result", "warnings", "error"}


async def assert_parity(
    model: onnx.ModelProto,
    inputs: dict[str, np.ndarray],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> None:
    expected = run_onnxruntime(model, inputs)
    got = await run_aimodel(model, inputs)
    assert len(expected) == len(got)
    for i, (e, g) in enumerate(zip(expected, got, strict=True)):
        e = narrow_array(e, context=f"output {i}")
        g = np.asarray(g)
        assert g.shape == e.shape, (
            f"output {i} shape mismatch: got {g.shape}, want {e.shape}"
        )
        assert g.dtype == e.dtype, (
            f"output {i} dtype mismatch: got {g.dtype}, want {e.dtype}"
        )
        if e.dtype.kind in "biu":
            np.testing.assert_array_equal(g, e, err_msg=f"output {i} mismatch")
        else:
            np.testing.assert_allclose(
                g.astype(np.float64),
                e.astype(np.float64),
                rtol=rtol,
                atol=atol,
                err_msg=f"output {i} mismatch",
            )
