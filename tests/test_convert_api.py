# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the coreai_onnx.convert() convenience function."""

import numpy as np
import onnx

import coreai_onnx

from .helpers import requires_coreai_runtime, single_op_model


@requires_coreai_runtime
async def test_convert_one_liner(tmp_path):
    """convert() wraps OnnxConverter end-to-end; asset file is created."""
    x = np.random.default_rng(0).random((3, 4)).astype(np.float32)
    model = single_op_model("Relu", {"x": x})

    program = coreai_onnx.convert(model)
    program.optimize()
    asset_path = tmp_path / "m.aimodel"
    program.save_asset(asset_path)

    assert asset_path.exists()


def test_convert_accepts_path(tmp_path):
    """convert() accepts a Path to a .onnx file and returns a non-None AIProgram."""
    x = np.random.default_rng(1).random((2, 3)).astype(np.float32)
    model = single_op_model("Relu", {"x": x})

    onnx_path = tmp_path / "model.onnx"
    onnx.save(model, str(onnx_path))

    program = coreai_onnx.convert(onnx_path)
    assert program is not None


@requires_coreai_runtime
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
