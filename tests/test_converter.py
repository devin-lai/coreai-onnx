# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx import ConversionError, UnsupportedOpError

from .helpers import assert_parity, requires_coreai_runtime, single_op_model


@requires_coreai_runtime
async def test_add_parity() -> None:
    model = single_op_model(
        "Add",
        {
            "a": np.random.rand(2, 3).astype(np.float32),
            "b": np.random.rand(2, 3).astype(np.float32),
        },
    )
    await assert_parity(
        model,
        {
            "a": np.random.rand(2, 3).astype(np.float32),
            "b": np.random.rand(2, 3).astype(np.float32),
        },
    )


@requires_coreai_runtime
async def test_add_with_initializer() -> None:
    rng = np.random.default_rng(0)
    model = single_op_model(
        "Add",
        {"a": rng.random((4,), dtype=np.float32)},
        initializers={"w": rng.random((4,), dtype=np.float32)},
    )
    await assert_parity(model, {"a": rng.random((4,), dtype=np.float32)})


def test_unsupported_op_aggregated_error() -> None:
    model = single_op_model("Det", {"x": np.eye(3, dtype=np.float32)})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(UnsupportedOpError, match="Det"):
        converter.to_coreai()


def test_custom_lowering_registration() -> None:
    converter = coreai_onnx.OnnxConverter()

    @converter.register_onnx_lowering("com.example::MyOp")
    def lower(values_map, node, loc):  # pragma: no cover - registration only
        raise NotImplementedError

    assert "com.example::MyOp" in converter._user_defined_lowering
    with pytest.raises(ValueError, match="reserved"):
        converter.register_onnx_lowering("coreai::X")


def test_empty_converter_raises() -> None:
    with pytest.raises(RuntimeError, match="add_onnx_model"):
        coreai_onnx.OnnxConverter().to_coreai()


def test_to_coreai_is_single_use_after_success() -> None:
    model = single_op_model("Relu", {"x": np.zeros((2, 3), dtype=np.float32)})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)

    converter.to_coreai()

    with pytest.raises(RuntimeError, match="only be called once"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Fix 1 — duplicate entrypoint names
# ---------------------------------------------------------------------------


def test_duplicate_entrypoint_raises() -> None:
    model = single_op_model("Det", {"x": np.eye(3, dtype=np.float32)})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model, entrypoint_name="main")
    with pytest.raises(ValueError, match="already staged"):
        converter.add_onnx_model(model, entrypoint_name="main")


# ---------------------------------------------------------------------------
# Fix 2 — result-count mismatch in _lower_node
# ---------------------------------------------------------------------------


def _make_two_output_custom_model() -> onnx.ModelProto:
    """Build a minimal model with a 2-output node in domain 'com.example'."""
    node = helper.make_node(
        "TwoOut",
        inputs=["x"],
        outputs=["out0", "out1"],
        domain="com.example",
    )
    graph = helper.make_graph(
        [node],
        "test_two_out",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("out0", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("out1", TensorProto.FLOAT, [2]),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("com.example", 1),
        ],
        ir_version=10,
    )


def test_result_count_mismatch_raises_conversion_error() -> None:
    model = _make_two_output_custom_model()
    converter = coreai_onnx.OnnxConverter()

    # Register a lowering that returns only one Value instead of the two required.
    from coreai._compiler.dialects import coreai as _coreai

    @converter.register_onnx_lowering("com.example::TwoOut")
    def lower_two_out(values_map, node, loc):  # pragma: no cover - error path
        # Return only a single constant — wrong: node has 2 required outputs.
        import numpy as _np

        return _coreai.constant(_np.array([1.0, 2.0], dtype=_np.float32))

    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="2 output") as exc_info:
        converter.to_coreai()
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_lowering_binds_results_to_nonempty_outputs() -> None:
    """A node with a gapped optional output ['A', '', 'C'] whose lowering returns
    one Value per non-empty output must bind A and C. Zipping results against the
    raw output list would spend a result on the '' slot and leave C unbound, so
    graph-output validation would reject this otherwise-valid model."""
    node = helper.make_node(
        "Gapped", inputs=["x"], outputs=["A", "", "C"], domain="com.example"
    )
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("C", TensorProto.FLOAT, [2]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("com.example", 1),
        ],
        ir_version=10,
    )
    converter = coreai_onnx.OnnxConverter()

    @converter.register_onnx_lowering("com.example::Gapped")
    def lower_gapped(values_map, node, loc):
        x = values_map[node.input[0]]
        return [x, x]  # one result per non-empty output (A, C)

    converter.add_onnx_model(model)
    converter.to_coreai()  # must not raise "graph output 'C' was not produced"


# ---------------------------------------------------------------------------
# Fix 3 — register_onnx_lowering: no silent overrides
# ---------------------------------------------------------------------------


def test_register_builtin_without_override_raises() -> None:
    converter = coreai_onnx.OnnxConverter()
    with pytest.raises(ValueError, match="already exists"):
        converter.register_onnx_lowering("Add")


def test_register_builtin_with_allow_override_succeeds() -> None:
    converter = coreai_onnx.OnnxConverter()

    @converter.register_onnx_lowering("Add", allow_override=True)
    def my_add(values_map, node, loc):  # pragma: no cover
        raise NotImplementedError

    assert converter._user_defined_lowering.get("Add") is my_add


def test_reserved_domain_still_rejected() -> None:
    converter = coreai_onnx.OnnxConverter()
    with pytest.raises(ValueError, match="reserved"):
        converter.register_onnx_lowering("ai.onnx::Add")


# ---------------------------------------------------------------------------
# converter hand-off contract: only a 4th positional param named 'converter'
# ---------------------------------------------------------------------------


def _custom_op_model() -> onnx.ModelProto:
    node = helper.make_node("MyOp", inputs=["x"], outputs=["y"], domain="com.example")
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 22),
            helper.make_opsetid("com.example", 1),
        ],
        ir_version=10,
    )


def test_defaulted_fourth_param_keeps_its_default() -> None:
    """A 4th param not named 'converter' (e.g. a tuning knob) must never be
    silently bound to the OnnxConverter instance."""
    from coreai._compiler.dialects import coreai as _coreai

    converter = coreai_onnx.OnnxConverter()
    captured = {}

    @converter.register_onnx_lowering("com.example::MyOp")
    def lower(values_map, node, loc, scale=2.0):
        captured["scale"] = scale
        return _coreai.mul(
            values_map["x"],
            _coreai.constant(np.full(2, float(scale), dtype=np.float32), loc=loc),
            loc=loc,
        )

    converter.add_onnx_model(_custom_op_model())
    converter.to_coreai()
    assert captured["scale"] == 2.0


def test_fourth_param_named_converter_receives_converter() -> None:
    from coreai._compiler.dialects import coreai as _coreai

    converter = coreai_onnx.OnnxConverter()
    captured = {}

    @converter.register_onnx_lowering("com.example::MyOp")
    def lower(values_map, node, loc, converter):
        captured["converter"] = converter
        return _coreai.mul(
            values_map["x"],
            _coreai.constant(np.ones(2, dtype=np.float32), loc=loc),
            loc=loc,
        )

    converter.add_onnx_model(_custom_op_model())
    converter.to_coreai()
    assert captured["converter"] is converter


def test_varargs_lowering_gets_three_args() -> None:
    from coreai._compiler.dialects import coreai as _coreai

    converter = coreai_onnx.OnnxConverter()
    captured = {}

    @converter.register_onnx_lowering("com.example::MyOp")
    def lower(*args):
        captured["nargs"] = len(args)
        values_map, _node, loc = args[:3]
        return _coreai.mul(
            values_map["x"],
            _coreai.constant(np.ones(2, dtype=np.float32), loc=loc),
            loc=loc,
        )

    converter.add_onnx_model(_custom_op_model())
    converter.to_coreai()
    assert captured["nargs"] == 3


# ---------------------------------------------------------------------------
# float16 models: warn that Core AI runtime load support is partial
# ---------------------------------------------------------------------------


def test_f16_graph_input_warns() -> None:
    model = single_op_model(
        "Add",
        {
            "a": np.zeros((2, 3), dtype=np.float16),
            "b": np.zeros((2, 3), dtype=np.float16),
        },
    )
    converter = coreai_onnx.OnnxConverter()
    with pytest.warns(UserWarning, match="float16"):
        converter.add_onnx_model(model)


def test_f16_initializer_warns() -> None:
    model = single_op_model(
        "Add",
        {"a": np.zeros((4,), dtype=np.float16)},
        initializers={"w": np.zeros((4,), dtype=np.float16)},
    )
    converter = coreai_onnx.OnnxConverter()
    with pytest.warns(UserWarning, match="float16"):
        converter.add_onnx_model(model)


def test_f32_model_does_not_warn_about_f16() -> None:
    import warnings

    model = single_op_model(
        "Add",
        {
            "a": np.zeros((2, 3), dtype=np.float32),
            "b": np.zeros((2, 3), dtype=np.float32),
        },
    )
    converter = coreai_onnx.OnnxConverter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converter.add_onnx_model(model)
    assert not [w for w in caught if "float16" in str(w.message)]


# ---------------------------------------------------------------------------
# error taxonomy: valid-model policy rejections raise CoreaiOnnxError subclasses
# ---------------------------------------------------------------------------


def test_initializer_overflow_raises_conversion_error() -> None:
    """int64 initializer outside int32 range: ConversionError, not OverflowError."""
    big = onnx.numpy_helper.from_array(np.array([1, 2**40], dtype=np.int64), name="big")
    node = helper.make_node("Add", ["X", "big"], ["Y"], name="add0")
    graph = helper.make_graph(
        [node],
        "g",
        inputs=[helper.make_tensor_value_info("X", TensorProto.INT64, [2])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.INT64, [2])],
        initializer=[big],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    with pytest.raises(ConversionError, match="big"):
        coreai_onnx.convert(model)


def test_unsupported_input_dtype_raises_conversion_error() -> None:
    """COMPLEX64 graph input: ConversionError, not a bare ValueError."""
    node = helper.make_node("Identity", ["X"], ["Y"], name="id0")
    graph = helper.make_graph(
        [node],
        "g",
        inputs=[helper.make_tensor_value_info("X", TensorProto.COMPLEX64, [2])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.COMPLEX64, [2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    with pytest.raises(ConversionError, match="COMPLEX64"):
        coreai_onnx.convert(model)


def test_add_onnx_model_does_not_mutate_caller_proto() -> None:
    """The preprocessing/fusion passes work on a copy: the caller's proto must
    stay byte-identical (e.g. so it can be fed to onnxruntime afterwards)."""
    node = helper.make_node("Identity", ["X"], ["mid"], name="id0")
    relu = helper.make_node("Relu", ["mid"], ["Y"], name="relu0")
    graph = helper.make_graph(
        [node, relu],
        "g",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    before = model.SerializeToString()
    coreai_onnx.convert(model)
    assert model.SerializeToString() == before
