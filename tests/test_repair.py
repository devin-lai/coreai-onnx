# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Automatic-repair engine: known-safe rewrites + the parity safety net.

The structural and ONNX-Runtime equivalence tests need no Core AI runtime; the
end-to-end parity test (the repaired model actually loads and runs on Core AI,
which the original float16 model cannot) is runtime-gated.
"""

import numpy as np
import onnx

from coreai_onnx._repair import apply_repairs
from coreai_onnx._repair._strategies import STRATEGIES

from .helpers import (
    assert_parity,
    coreai_runtime_test,
    run_onnxruntime,
    single_op_model,
)

F16 = onnx.TensorProto.FLOAT16
F32 = onnx.TensorProto.FLOAT


def _all_elem_types(model: onnx.ModelProto) -> set[int]:
    g = model.graph
    types = {vi.type.tensor_type.elem_type for vi in g.input}
    types |= {vi.type.tensor_type.elem_type for vi in g.output}
    types |= {vi.type.tensor_type.elem_type for vi in g.value_info}
    types |= {init.data_type for init in g.initializer}
    return types


def _fp16_mul_model() -> onnx.ModelProto:
    """x (f16 input) * w (f16 initializer) — exercises input + initializer + op."""
    x = np.zeros((2, 3), dtype=np.float16)
    w = np.ones((3,), dtype=np.float16)
    return single_op_model("Mul", {"x": x}, initializers={"w": w})


def test_apply_repairs_promotes_every_float16_to_float32():
    model = _fp16_mul_model()
    assert F16 in _all_elem_types(model)  # precondition

    repaired, records = apply_repairs(model)

    assert F16 not in _all_elem_types(repaired), "no float16 must remain"
    assert repaired.graph.input[0].type.tensor_type.elem_type == F32
    onnx.checker.check_model(repaired)
    assert len(records) == 1
    assert records[0].name == "promote_float16_to_float32"
    assert records[0].details == {"inputs": ["x"]}


def test_apply_repairs_does_not_mutate_input():
    model = _fp16_mul_model()
    repaired, _ = apply_repairs(model)
    assert model.graph.input[0].type.tensor_type.elem_type == F16, "original untouched"
    assert repaired is not model


def test_apply_repairs_noop_on_float32_model():
    model = single_op_model("Relu", {"x": np.zeros((4,), dtype=np.float32)})
    repaired, records = apply_repairs(model)
    assert records == []
    assert repaired.SerializeToString() == model.SerializeToString()


def test_promotion_preserves_results_within_float16_precision():
    """The promoted float32 graph must match the original float16 graph on
    ONNX Runtime to within float16 precision (no Core AI runtime needed)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3)).astype(np.float16)
    w = rng.standard_normal((3,)).astype(np.float16)
    model = single_op_model("Mul", {"x": x}, initializers={"w": w})

    expected = run_onnxruntime(model, {"x": x})[0]
    repaired, _ = apply_repairs(model)
    got = run_onnxruntime(repaired, {"x": x.astype(np.float32)})[0]

    assert expected.dtype == np.float16
    assert got.dtype == np.float32
    np.testing.assert_allclose(
        got.astype(np.float64),
        expected.astype(np.float64),
        rtol=1e-3,
        atol=1e-3,
    )


def test_strategies_are_well_formed():
    for s in STRATEGIES:
        assert s.name
        assert s.summary
        assert callable(s.detect)
        assert callable(s.apply)


@coreai_runtime_test
async def test_repaired_fp16_model_converts_and_runs_on_core_ai():
    """End-to-end: a float16 model the Core AI runtime cannot load is repaired,
    then converts and runs with parity vs ONNX Runtime."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 3)).astype(np.float16)
    w = rng.standard_normal((3,)).astype(np.float16)
    model = single_op_model("Mul", {"x": x}, initializers={"w": w})

    repaired, records = apply_repairs(model)
    assert records, "repair should have applied to a float16 model"
    await assert_parity(repaired, {"x": x.astype(np.float32)})
