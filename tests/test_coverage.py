# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_onnx._coverage (no runtime required)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from coreai_onnx._coverage import (
    analyze,
    coverage_markdown,
    supported_ops,
)
from tests.helpers import single_op_model

# ---------------------------------------------------------------------------
# supported_ops
# ---------------------------------------------------------------------------


def test_supported_ops_contains_known_set():
    ops = supported_ops()
    required = {"Add", "Conv", "Reshape", "Softmax", "If", "MatMul", "Resize"}
    assert required <= ops, f"Missing ops: {required - ops}"


# ---------------------------------------------------------------------------
# analyze - unsupported op
# ---------------------------------------------------------------------------


def test_analyze_unsupported_op():
    # Det is not in the resolver
    model = single_op_model(
        "Det",
        {"X": np.zeros((3, 3), dtype=np.float32)},
        n_outputs=1,
    )
    report = analyze(model)
    assert report.total_nodes == 1
    assert report.unsupported == {"Det": 1}
    assert not report.convertible


# ---------------------------------------------------------------------------
# analyze - supported op
# ---------------------------------------------------------------------------


def test_analyze_supported_op():
    model = single_op_model(
        "Relu",
        {"X": np.zeros((4,), dtype=np.float32)},
        n_outputs=1,
    )
    report = analyze(model)
    assert report.convertible
    assert report.op_histogram == {"Relu": 1}


# ---------------------------------------------------------------------------
# analyze - recursion into If branches
# ---------------------------------------------------------------------------


def _if_model_with_det_in_branch() -> onnx.ModelProto:
    """Build an If model whose then-branch contains a Det node (unsupported)."""
    # then-branch: Det(x) -> shape [1] via a workaround - just use Det on a
    # 3x3 input; branch output type must match else-branch.
    then_g = helper.make_graph(
        [
            helper.make_node("Det", ["x"], ["then_out"]),
        ],
        "then_g",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [])],
    )
    else_g = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["else_out"],
                value=numpy_helper.from_array(np.array(0.0, dtype=np.float32)),
            ),
        ],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [])],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        "test_if_with_det",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )


def test_analyze_recurses_into_if_branches():
    model = _if_model_with_det_in_branch()
    report = analyze(model)
    # Outer graph: 1 If node; then-branch: 1 Det; else-branch: 1 Constant
    assert report.total_nodes == 3
    # Det is unsupported; it lives inside the branch
    assert "Det" in report.unsupported
    assert report.unsupported["Det"] == 1
    assert not report.convertible


# ---------------------------------------------------------------------------
# coverage_markdown
# ---------------------------------------------------------------------------


def test_coverage_markdown_header_and_add_row():
    md = coverage_markdown()
    assert "| Op key | Status |" in md
    assert "| Add | supported |" in md
