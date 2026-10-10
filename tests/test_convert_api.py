# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the coreai_onnx.convert() convenience function."""

import numpy as np
import onnx
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx._ir import mlir_module_of

from .helpers import COREAI_CONVERSION_MARKS, coreai_runtime_test, single_op_model

pytestmark = [*COREAI_CONVERSION_MARKS]


@coreai_runtime_test
async def test_convert_one_liner(tmp_path):
    """convert() wraps OnnxConverter end-to-end; asset file is created."""
    x = np.random.default_rng(0).random((3, 4)).astype(np.float32)
    model = single_op_model("Relu", {"x": x})

    program = coreai_onnx.convert(model)
    asset_path = tmp_path / "m.aimodel"
    program.save_asset(asset_path)

    assert asset_path.exists()


def _op_names(program) -> list[str]:
    names: list[str] = []

    def walk(op):
        names.append(op.name)
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner)

    for op in mlir_module_of(program).body.operations:
        walk(op)
    return names


def _foldable_model() -> onnx.ModelProto:
    """Cast(Cast(x)): a chain the pre-compilation rewrite fuses into one cast."""
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [3])
    nodes = [
        helper.make_node("Cast", ["x"], ["f32"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["f32"], ["y"], to=TensorProto.FLOAT16),
    ]
    graph = helper.make_graph(nodes, "casts", [x], [y])
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )


def test_convert_returns_optimized_program_by_default():
    """convert() applies Core AI's pre-compilation rewrite: the si32->f32->f16
    cast chain is fused into one cast, with no separate optimize() step."""
    optimized = coreai_onnx.convert(_foldable_model())
    assert _op_names(optimized).count("coreai.cast") == 1


def test_convert_optimize_false_keeps_lowered_ir():
    """optimize=False returns the IR exactly as the lowerings emitted it."""
    raw = coreai_onnx.convert(_foldable_model(), optimize=False)
    assert _op_names(raw).count("coreai.cast") == 2


def test_to_coreai_optimize_keyword():
    """OnnxConverter.to_coreai mirrors convert()'s optimize switch."""
    counts = {}
    for optimize in (True, False):
        converter = coreai_onnx.OnnxConverter()
        converter.add_onnx_model(_foldable_model())
        program = converter.to_coreai(optimize=optimize)
        counts[optimize] = _op_names(program).count("coreai.cast")
    assert counts == {True: 1, False: 2}


def test_convert_accepts_path(tmp_path):
    """convert() accepts a Path to a .onnx file and returns a non-None AIProgram."""
    x = np.random.default_rng(1).random((2, 3)).astype(np.float32)
    model = single_op_model("Relu", {"x": x})

    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, str(onnx_path))

    program = coreai_onnx.convert(onnx_path)
    assert program is not None


@coreai_runtime_test
async def test_convert_forwards_kwargs(tmp_path):
    """output_names override propagates: output dict key matches the renamed name."""
    from coreai.runtime import NDArray

    x = np.random.default_rng(2).random((2, 3)).astype(np.float32)
    model = single_op_model("Relu", {"x": x})

    program = coreai_onnx.convert(model, input_names=["x"], output_names=["renamed"])
    asset = program.save_asset(tmp_path / "m.aimodel")
    async with asset.executable() as ai_model:
        fn = ai_model.load_function("main")
        out = await fn({"x": NDArray(x)})

    assert "renamed" in out
