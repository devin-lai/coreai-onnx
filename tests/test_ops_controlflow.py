# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for the If control-flow op lowering (Task 13).

If models cannot be expressed with ``single_op_model`` (branches are GraphProto
attributes), so models are built with ``onnx.helper`` directly. onnxruntime
executes If natively, providing the reference for both branch outcomes.
"""

import zlib

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from .helpers import COREAI_RUNTIME_MARKS, assert_parity, requires_coreai_runtime

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


def _f32_const(name: str, out: str, value: float):
    return helper.make_node(
        "Constant",
        [],
        [out],
        name=name,
        value=numpy_helper.from_array(np.array(value, dtype=np.float32)),
    )


def _make_model(graph):
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )


# ---------------------------------------------------------------------------
# Basic If: then Add(x, 1) / else Mul(x, 2); x captured from the outer graph.
#
# The cond is a bool GRAPH INPUT fed straight into coreai.if_. (Known runtime
# quirk: bool graph inputs crash *some* primitives — not_, and_/or_/xor — but
# coreai.if_ consumes them fine, so no Greater-derived workaround is needed.)
# ---------------------------------------------------------------------------


def _basic_if_model():
    then_g = helper.make_graph(
        [
            _f32_const("c_one", "one", 1.0),
            helper.make_node("Add", ["x", "one"], ["then_out"]),
        ],
        "then_g",
        [],  # If branches take no inputs; x is captured by name
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])],
    )
    else_g = helper.make_graph(
        [
            _f32_const("c_two", "two", 2.0),
            helper.make_node("Mul", ["x", "two"], ["else_out"]),
        ],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        "test_if_basic",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )
    return _make_model(graph)


async def test_if_basic():
    rng = np.random.default_rng(_seed("if-basic"))
    x = rng.random(3).astype(np.float32)
    model = _basic_if_model()
    await assert_parity(model, {"x": x, "cond": np.array(True)})
    await assert_parity(model, {"x": x, "cond": np.array(False)})


# ---------------------------------------------------------------------------
# Branch subgraph with its own initializer
# ---------------------------------------------------------------------------


async def test_if_with_initializer_in_branch():
    rng = np.random.default_rng(_seed("if-branch-init"))
    x = rng.random(4).astype(np.float32)
    bias = rng.random(4).astype(np.float32)

    then_g = helper.make_graph(
        [helper.make_node("Add", ["x", "bias"], ["then_out"])],
        "then_g",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [4])],
        initializer=[numpy_helper.from_array(bias, name="bias")],
    )
    else_g = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["else_out"])],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [4])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        "test_if_branch_init",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
    )
    model = _make_model(graph)
    await assert_parity(model, {"x": x, "cond": np.array(True)})
    await assert_parity(model, {"x": x, "cond": np.array(False)})


# ---------------------------------------------------------------------------
# Nested If: inner If inside the then-branch of the outer If
# ---------------------------------------------------------------------------


def _nested_if_model():
    inner_then = helper.make_graph(
        [
            _f32_const("c_one", "one", 1.0),
            helper.make_node("Add", ["x", "one"], ["inner_then_out"]),
        ],
        "inner_then",
        [],
        [helper.make_tensor_value_info("inner_then_out", TensorProto.FLOAT, [3])],
    )
    inner_else = helper.make_graph(
        [
            _f32_const("c_hundred", "hundred", 100.0),
            helper.make_node("Add", ["x", "hundred"], ["inner_else_out"]),
        ],
        "inner_else",
        [],
        [helper.make_tensor_value_info("inner_else_out", TensorProto.FLOAT, [3])],
    )
    # then-branch of the OUTER If contains the inner If; cond2 is captured
    # from the outer graph two region levels up.
    outer_then = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["cond2"],
                ["outer_then_out"],
                then_branch=inner_then,
                else_branch=inner_else,
            )
        ],
        "outer_then",
        [],
        [helper.make_tensor_value_info("outer_then_out", TensorProto.FLOAT, [3])],
    )
    outer_else = helper.make_graph(
        [
            _f32_const("c_two", "two", 2.0),
            helper.make_node("Mul", ["x", "two"], ["outer_else_out"]),
        ],
        "outer_else",
        [],
        [helper.make_tensor_value_info("outer_else_out", TensorProto.FLOAT, [3])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond1"], ["y"], then_branch=outer_then, else_branch=outer_else
            )
        ],
        "test_if_nested",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("cond1", TensorProto.BOOL, []),
            helper.make_tensor_value_info("cond2", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3])],
    )
    return _make_model(graph)


async def test_if_nested():
    rng = np.random.default_rng(_seed("if-nested"))
    x = rng.random(3).astype(np.float32)
    model = _nested_if_model()
    # three reachable paths: (T,T) -> x+1, (T,F) -> x+100, (F,*) -> x*2
    t, f = np.array(True), np.array(False)
    await assert_parity(model, {"x": x, "cond1": t, "cond2": t})
    await assert_parity(model, {"x": x, "cond1": t, "cond2": f})
    await assert_parity(model, {"x": x, "cond1": f, "cond2": t})


# ---------------------------------------------------------------------------
# Multiple outputs per branch
# ---------------------------------------------------------------------------


def _multi_output_if_model():
    then_g = helper.make_graph(
        [
            _f32_const("c_one", "one", 1.0),
            helper.make_node("Add", ["x", "one"], ["then_a"]),
            helper.make_node("Neg", ["x"], ["then_b"]),
        ],
        "then_g",
        [],
        [
            helper.make_tensor_value_info("then_a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("then_b", TensorProto.FLOAT, [2, 3]),
        ],
    )
    else_g = helper.make_graph(
        [
            _f32_const("c_two", "two", 2.0),
            helper.make_node("Mul", ["x", "two"], ["else_a"]),
            helper.make_node("Abs", ["x"], ["else_b"]),
        ],
        "else_g",
        [],
        [
            helper.make_tensor_value_info("else_a", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("else_b", TensorProto.FLOAT, [2, 3]),
        ],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["y0", "y1"], then_branch=then_g, else_branch=else_g
            )
        ],
        "test_if_multi_out",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [
            helper.make_tensor_value_info("y0", TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, [2, 3]),
        ],
    )
    return _make_model(graph)


def test_if_multiple_outputs_converts():
    """Multi-result If converts and serializes correctly (IR-level check).

    Execution is covered by the skipped test below: the Core AI runtime's
    JIT specialization hangs (100% CPU, never returns) on an If op with
    multiple results. Conversion and ``save_asset`` are unaffected.
    """
    import tempfile
    from pathlib import Path

    import coreai_onnx

    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(_multi_output_if_model())
    program = converter.to_coreai()
    with tempfile.TemporaryDirectory() as td:
        program.save_asset(Path(td) / "m.aimodel")


@pytest.mark.skip(
    reason="Core AI runtime hangs (100% CPU in JIT specialization) when "
    "executing an If op with multiple results; the IR converts and saves "
    "fine — see test_if_multiple_outputs_converts. Upstream coreai-core issue."
)
async def test_if_multiple_outputs():
    rng = np.random.default_rng(_seed("if-multi-out"))
    x = rng.random((2, 3)).astype(np.float32)
    model = _multi_output_if_model()
    await assert_parity(model, {"x": x, "cond": np.array(True)})
    await assert_parity(model, {"x": x, "cond": np.array(False)})


# ---------------------------------------------------------------------------
# Branches with legally differing output shapes (valid ONNX, dynamic node
# output). coreai.if requires both regions to yield exactly the declared
# result type, so this must fail with a clear error at conversion time —
# not an opaque save_asset failure long after.
# ---------------------------------------------------------------------------


def test_if_mismatched_branch_shapes_rejected():
    import coreai_onnx
    from coreai_onnx.errors import ConversionError

    def const_branch(name, arr, out):
        return helper.make_graph(
            [
                helper.make_node(
                    "Constant", [], [out], value=numpy_helper.from_array(arr)
                )
            ],
            name,
            [],
            [helper.make_tensor_value_info(out, TensorProto.FLOAT, [len(arr)])],
        )

    then_g = const_branch("then_g", np.array([1.0, 2.0], dtype=np.float32), "t_out")
    else_g = const_branch(
        "else_g", np.array([3.0, 4.0, 5.0], dtype=np.float32), "e_out"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        "test_if_shape_mismatch",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(_make_model(graph))
    with pytest.raises(ConversionError, match="shape"):
        converter.to_coreai()
