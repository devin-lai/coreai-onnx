# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MLIR-output tests: assert on the textual IR of converted programs.

These run on any platform — they convert models but never execute them.
"""

import numpy as np
import pytest

import coreai_onnx
from coreai_onnx.errors import UnsupportedOpError

from .helpers import single_op_model

pytestmark = pytest.mark.ir


def test_relu_emits_graph_and_relu():
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model("Relu", {"x": x})
    program = coreai_onnx.convert(model)
    text = str(program)
    assert "coreai.graph" in text
    assert "relu" in text


def test_conv_emits_conv():
    x = np.zeros((1, 1, 4, 4), dtype=np.float32)
    w = np.zeros((1, 1, 3, 3), dtype=np.float32)
    model = single_op_model("Conv", {"x": x}, initializers={"w": w})
    program = coreai_onnx.convert(model)
    assert "conv" in str(program)


def test_default_entrypoint_is_main():
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model("Relu", {"x": x})
    program = coreai_onnx.convert(model)
    assert "@main" in str(program)


def test_custom_entrypoint_name():
    x = np.zeros((2, 3), dtype=np.float32)
    model = single_op_model("Relu", {"x": x})
    program = coreai_onnx.convert(model, entrypoint_name="predict")
    text = str(program)
    assert "@predict" in text
    assert "@main" not in text


def test_initializer_becomes_constant():
    x = np.zeros((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    model = single_op_model("Add", {"x": x}, initializers={"b": b})
    assert "coreai.constant" in str(coreai_onnx.convert(model))


def test_static_float_concat_avoids_native_concat_and_preserves_signed_zero():
    a = np.zeros((1, 2, 4, 4), dtype=np.float32)
    b = np.zeros((1, 3, 4, 4), dtype=np.float32)
    c = np.zeros((1, 1, 4, 4), dtype=np.float32)
    model = single_op_model("Concat", {"a": a, "b": b, "c": c}, attrs={"axis": 1})

    text = str(coreai_onnx.convert(model))

    assert "coreai.concat" not in text
    assert "coreai.slice_update" not in text
    assert "coreai.pad" in text
    assert "broadcasting_add" not in text
    assert "broadcasting_where" in text


def test_static_integer_concat_uses_pad_add_barrier():
    a = np.zeros((1, 2, 4, 4), dtype=np.int32)
    b = np.zeros((1, 3, 4, 4), dtype=np.int32)
    c = np.zeros((1, 1, 4, 4), dtype=np.int32)
    model = single_op_model("Concat", {"a": a, "b": b, "c": c}, attrs={"axis": 1})

    text = str(coreai_onnx.convert(model))

    assert "coreai.concat" not in text
    assert "coreai.slice_update" not in text
    assert "coreai.pad" in text
    assert "broadcasting_add" in text
    assert "broadcasting_where" not in text


def test_linear_resize_avoids_native_interpolate():
    """Linear Resize must lower to compile-time gather+lerp, not
    coreai.interpolate: the native resample kernel (MPSNDArrayResample)
    aborts the process at GPU/ANE compile time in real decoder graphs
    ('source and destination channels mismatch', track_anything segment),
    and barriers around the interpolate do not prevent the bad fusion."""
    x = np.zeros((1, 3, 8, 8), dtype=np.float32)
    model = single_op_model(
        "Resize",
        {"x": x},
        attrs={"mode": "linear", "coordinate_transformation_mode": "half_pixel"},
        initializers={
            "roi": np.array([], dtype=np.float32),
            "scales": np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32),
        },
    )

    text = str(coreai_onnx.convert(model))

    assert "coreai.interpolate" not in text


def test_prelu_avoids_predicate_where():
    x = np.zeros((1, 3, 4, 4), dtype=np.float32)
    slope = np.full((1, 3, 1, 1), 0.25, dtype=np.float32)
    model = single_op_model("PRelu", {"x": x}, initializers={"slope": slope})

    text = str(coreai_onnx.convert(model))

    assert "coreai.relu" in text
    assert "broadcasting_mul" in text
    assert "broadcasting_sub" in text
    assert "broadcasting_greater" not in text
    assert "broadcasting_where" not in text


def test_unsupported_op_raises_before_emission():
    x = np.zeros((3, 3), dtype=np.float32)
    model = single_op_model("Det", {"x": x})
    with pytest.raises(UnsupportedOpError, match="Det"):
        coreai_onnx.convert(model)
