# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import warnings
from contextlib import suppress

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from coreai_onnx._fusion import fuse_attention
from coreai_onnx._passes import (
    BASELINE_OPSET,
    eliminate_dead_nodes,
    fold_constants,
    preprocess,
    remove_identity,
)

from .helpers import assert_parity, coreai_runtime_test, coreai_test, run_onnxruntime


def _model(nodes, inputs, outputs, initializers=(), opset=22):
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializers)
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )


def test_remove_identity():
    m = _model(
        [
            helper.make_node("Identity", ["x"], ["t"]),
            helper.make_node("Relu", ["t"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    m = remove_identity(m)
    assert [n.op_type for n in m.graph.node] == ["Relu"]
    assert m.graph.node[0].input[0] == "x"


def test_identity_feeding_graph_output_is_kept_correct():
    m = _model(
        [helper.make_node("Identity", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    m = remove_identity(m)
    # Identity feeding a graph output cannot be rewired away — it stays
    assert [n.op_type for n in m.graph.node] == ["Identity"]


def test_eliminate_dead_nodes():
    m = _model(
        [
            helper.make_node("Relu", ["x"], ["y"]),
            helper.make_node("Sigmoid", ["x"], ["unused"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    m = eliminate_dead_nodes(m)
    assert [n.op_type for n in m.graph.node] == ["Relu"]


def test_fold_constants_shape_chain():
    # Shape(x) → Gather → Unsqueeze → Concat feeding Reshape: folds to initializer
    m = _model(
        [
            helper.make_node("Shape", ["x"], ["s"]),
            helper.make_node("Gather", ["s", "idx"], ["d0"], axis=0),
            helper.make_node("Unsqueeze", ["d0", "ax"], ["d0u"]),
            helper.make_node("Concat", ["d0u", "rest"], ["newshape"], axis=0),
            helper.make_node("Reshape", ["x", "newshape"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers=[
            onnx.numpy_helper.from_array(np.array(0, dtype=np.int64), "idx"),
            onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), "ax"),
            onnx.numpy_helper.from_array(np.array([-1], dtype=np.int64), "rest"),
        ],
    )
    m = fold_constants(m)
    assert [n.op_type for n in m.graph.node] == ["Reshape"]
    assert {i.name for i in m.graph.initializer} >= {"newshape"}


def test_preprocess_upgrades_old_opset():
    m = _model(
        [helper.make_node("Relu", ["x"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        opset=11,
    )
    m = preprocess(m)
    (dom,) = [o for o in m.opset_import if o.domain == ""]
    assert dom.version >= BASELINE_OPSET


def test_preprocess_rejects_invalid_model():
    # Node references an undeclared input — checker must reject this.
    node = helper.make_node("Relu", ["undeclared"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    m = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 22)], ir_version=10
    )
    import pytest

    with pytest.raises(onnx.checker.ValidationError):
        preprocess(m)


# ---------------------------------------------------------------------------
# C1 — remove_identity must rewrite subgraph node inputs
# ---------------------------------------------------------------------------


def _make_if_model_with_identity():
    """Identity x→t; an If node whose then/else branches consume t."""
    # then-branch: node consuming 't', producing branch output
    then_relu = helper.make_node("Relu", ["t"], ["then_out"])
    then_graph = helper.make_graph(
        [then_relu],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    # else-branch: node also consuming 't'
    else_relu = helper.make_node("Relu", ["t"], ["else_out"])
    else_graph = helper.make_graph(
        [else_relu],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    identity_node = helper.make_node("Identity", ["x"], ["t"])
    graph = helper.make_graph(
        [identity_node, if_node],
        "g",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )


def test_remove_identity_rewrites_subgraph_inputs():
    """After removing Identity x→t, subgraph nodes consuming 't' must use 'x'."""
    m = _make_if_model_with_identity()
    m = remove_identity(m)
    # The Identity node must be gone
    assert all(n.op_type != "Identity" for n in m.graph.node)
    # Find the If node and inspect its branch subgraphs
    if_node = next(n for n in m.graph.node if n.op_type == "If")
    for attr in if_node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            for sn in attr.g.node:
                assert "t" not in sn.input, (
                    f"subgraph node {sn.op_type} still consumes renamed value 't'"
                )
                assert "x" in sn.input


def test_remove_identity_does_not_rewrite_shadowed_subgraph_name():
    """A subgraph that rebinds the renamed name (here a branch-local initializer
    't' shadowing an outer dead Identity x→t) must keep referencing its own 't';
    rewriting it to 'x' would silently miscompile a checker-valid model."""
    t_init = numpy_helper.from_array(np.array([5, 7], dtype=np.float32), name="t")
    then_graph = helper.make_graph(
        [helper.make_node("Add", ["t", "t"], ["then_out"])],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
        initializer=[t_init],
    )
    e_init = numpy_helper.from_array(np.array([0, 0], dtype=np.float32), name="e2")
    else_graph = helper.make_graph(
        [helper.make_node("Add", ["e2", "e2"], ["else_out"])],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
        initializer=[e_init],
    )
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["t"]),  # dead in the outer scope
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_graph, else_branch=else_graph
            ),
        ],
        "g",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    m = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    onnx.checker.check_model(m)

    feed = {"cond": np.array(True), "x": np.array([1, 1], dtype=np.float32)}
    before = run_onnxruntime(m, feed)[0]

    cleaned = remove_identity(m)
    onnx.checker.check_model(cleaned)
    after = run_onnxruntime(cleaned, feed)[0]

    # Semantics preserved: then-branch is still Add(t, t) == [10, 14], not Add(x, x).
    np.testing.assert_array_equal(before, [10.0, 14.0])
    np.testing.assert_array_equal(after, before)
    if_node = next(n for n in cleaned.graph.node if n.op_type == "If")
    tb = next(a.g for a in if_node.attribute if a.name == "then_branch")
    assert list(tb.node[0].input) == ["t", "t"]


# ---------------------------------------------------------------------------
# C2 — fold_constants must keep folded values captured by subgraphs
# ---------------------------------------------------------------------------


def _make_constant_in_subgraph_model():
    """A Constant node whose output is only used inside an If branch."""
    # The Constant is in the main graph, but only consumed inside the If branch.
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const_val"],
        value=numpy_helper.from_array(np.array([1.0], dtype=np.float32)),
    )
    # then-branch adds const_val to x
    then_add = helper.make_node("Add", ["x", "const_val"], ["then_out"])
    then_graph = helper.make_graph(
        [then_add],
        "then_branch",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_relu = helper.make_node("Relu", ["x"], ["else_out"])
    else_graph = helper.make_graph(
        [else_relu],
        "else_branch",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    graph = helper.make_graph(
        [const_node, if_node],
        "g",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )


def test_fold_constants_keeps_value_used_in_subgraph():
    """A Constant consumed only inside an If branch must become an initializer."""
    m = _make_constant_in_subgraph_model()
    m = fold_constants(m)
    init_names = {i.name for i in m.graph.initializer}
    assert "const_val" in init_names, (
        "const_val was folded but not emitted as initializer despite subgraph use"
    )


# ---------------------------------------------------------------------------
# C3 — eliminate_dead_nodes must scan nested subgraphs
# ---------------------------------------------------------------------------


def _make_nested_if_model():
    """Add in main graph; If node whose inner-branch captures 'outer_val'."""
    # outer_val = Add(x, x) in the main graph
    add_node = helper.make_node("Add", ["x", "x"], ["outer_val"])

    # inner then-branch uses outer_val
    inner_add = helper.make_node("Add", ["outer_val", "x"], ["inner_out"])
    inner_then_graph = helper.make_graph(
        [inner_add],
        "inner_then",
        [],
        [helper.make_tensor_value_info("inner_out", TensorProto.FLOAT, [2])],
    )
    inner_relu = helper.make_node("Relu", ["x"], ["inner_else_out"])
    inner_else_graph = helper.make_graph(
        [inner_relu],
        "inner_else",
        [],
        [helper.make_tensor_value_info("inner_else_out", TensorProto.FLOAT, [2])],
    )
    inner_if = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["inner_y"],
        then_branch=inner_then_graph,
        else_branch=inner_else_graph,
    )
    # outer then-branch contains the inner If
    outer_then_graph = helper.make_graph(
        [inner_if],
        "outer_then",
        [],
        [helper.make_tensor_value_info("inner_y", TensorProto.FLOAT, [2])],
    )
    outer_relu = helper.make_node("Relu", ["x"], ["outer_else_out"])
    outer_else_graph = helper.make_graph(
        [outer_relu],
        "outer_else",
        [],
        [helper.make_tensor_value_info("outer_else_out", TensorProto.FLOAT, [2])],
    )
    outer_if = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["y"],
        then_branch=outer_then_graph,
        else_branch=outer_else_graph,
    )
    graph = helper.make_graph(
        [add_node, outer_if],
        "g",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )


def test_eliminate_dead_nodes_scans_nested_subgraphs():
    """Add producing outer_val must survive even if only consumed inside nested If."""
    m = _make_nested_if_model()
    m = eliminate_dead_nodes(m)
    op_types = [n.op_type for n in m.graph.node]
    assert "Add" in op_types, (
        "Add (producing outer_val) was eliminated despite being captured by nested subgraph"
    )


# ---------------------------------------------------------------------------
# C4 — Constant fast-path must handle non-tensor (value_float) variants
# ---------------------------------------------------------------------------


def _make_value_float_constant_model():
    """Constant with value_float attribute consumed by a Relu."""
    const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["c"],
        value_float=3.14,
    )
    relu_node = helper.make_node("Relu", ["c"], ["y"])
    graph = helper.make_graph(
        [const_node, relu_node],
        "g",
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )


def test_fold_constants_value_float_does_not_crash():
    """fold_constants must not raise on a Constant with value_float attribute."""
    m = _make_value_float_constant_model()
    # Must not raise; constant should either be folded or left intact
    m = fold_constants(m)
    # Either an initializer named 'c' exists, or a Constant node with output 'c' remains
    init_names = {i.name for i in m.graph.initializer}
    node_outputs = {o for n in m.graph.node for o in n.output}
    assert "c" in init_names or "c" in node_outputs, (
        "'c' was neither folded nor kept after fold_constants"
    )


# ---------------------------------------------------------------------------
# I1 — opset > MAX_KNOWN_OPSET warning must point at preprocess()'s caller
# ---------------------------------------------------------------------------


def _make_high_opset_model():
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 30)], ir_version=10
    )


def test_normalize_opset_warning_stacklevel():
    """When called via preprocess(), the warning must point at preprocess()'s
    caller (this test file), not at a line inside the _passes package."""
    m = _make_high_opset_model()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with suppress(Exception):
            preprocess(m)
    assert len(w) >= 1
    filename = w[0].filename
    assert "coreai_onnx/_passes" not in filename.replace("\\", "/"), (
        f"Warning points at {filename!r}; expected caller's file, not the internal module"
    )


# ---------------------------------------------------------------------------
# fold_constants size cap — giant constant expansions must stay runtime nodes
# ---------------------------------------------------------------------------


def test_fold_constants_caps_giant_results():
    # ConstantOfShape would expand 32 bytes of shape data into a ~17 MB dense
    # mask (the causal-attention-mask construction pattern); folding it bloats
    # the proto and bakes the mask into the asset.
    m = _model(
        [
            helper.make_node("ConstantOfShape", ["shp"], ["z"]),
            helper.make_node("Add", ["z", "one"], ["zm"]),
            helper.make_node("Add", ["x", "zm"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2100, 2048])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2100, 2048])],
        initializers=[
            numpy_helper.from_array(
                np.array([1, 1, 2100, 2048], dtype=np.int64), "shp"
            ),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), "one"),
        ],
    )
    m = fold_constants(m)
    assert [n.op_type for n in m.graph.node] == ["ConstantOfShape", "Add", "Add"]
    assert all(i.ByteSize() < 1 << 24 for i in m.graph.initializer)


def test_fold_constants_skips_giant_inputs():
    # A foldable op over an already-huge constant (e.g. Cast of a big weight)
    # must not be evaluated: the fold would duplicate the weight in the proto.
    m = _model(
        [
            helper.make_node("Cast", ["w"], ["w16"], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", ["x", "w16"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [4, 2200])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [4, 2048])],
        initializers=[
            numpy_helper.from_array(np.zeros((2200, 2048), dtype=np.float32), "w")
        ],
    )
    m = fold_constants(m)
    assert [n.op_type for n in m.graph.node] == ["Cast", "MatMul"]


def test_fold_constants_emits_folded_graph_output():
    m = _model(
        [helper.make_node("Add", ["c1", "c2"], ["y"])],
        [],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        initializers=[
            numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "c1"),
            numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), "c2"),
        ],
    )
    m = fold_constants(m)
    assert not m.graph.node
    (y,) = [i for i in m.graph.initializer if i.name == "y"]
    np.testing.assert_array_equal(numpy_helper.to_array(y), [4.0, 6.0])


# ---------------------------------------------------------------------------
# fold_constants — extended foldable set (mask/shape pipelines)
# ---------------------------------------------------------------------------


def test_fold_constants_extended_mask_pipeline():
    # Where(Equal(m, 0), -1e9, 0) over a constant padding mask — the additive
    # attention-mask construction emitted by transformer exporters.
    m = _model(
        [
            helper.make_node("Equal", ["pad", "zero_i"], ["eq"]),
            helper.make_node("Where", ["eq", "neg", "zero_f"], ["amask"]),
            helper.make_node("Add", ["x", "amask"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializers=[
            numpy_helper.from_array(np.array([[1, 0, 1, 0]], dtype=np.int64), "pad"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_i"),
            numpy_helper.from_array(np.array(-1e9, dtype=np.float32), "neg"),
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "zero_f"),
        ],
    )
    m = fold_constants(m)
    assert [n.op_type for n in m.graph.node] == ["Add"]
    (amask,) = [i for i in m.graph.initializer if i.name == "amask"]
    np.testing.assert_array_equal(
        numpy_helper.to_array(amask), [[0.0, -1e9, 0.0, -1e9]]
    )


def test_fold_constants_transpose_and_expand():
    m = _model(
        [
            helper.make_node("Transpose", ["w"], ["wt"], perm=[1, 0]),
            helper.make_node("Expand", ["c", "eshape"], ["e"]),
            helper.make_node("MatMul", ["x", "wt"], ["mm"]),
            helper.make_node("Add", ["mm", "e"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [5, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [5, 2])],
        initializers=[
            numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(2, 3), "w"),
            numpy_helper.from_array(np.array([7.0], dtype=np.float32), "c"),
            numpy_helper.from_array(np.array([5, 2], dtype=np.int64), "eshape"),
        ],
    )
    m = fold_constants(m)
    assert [n.op_type for n in m.graph.node] == ["MatMul", "Add"]
    assert {i.name for i in m.graph.initializer} >= {"wt", "e"}


# ---------------------------------------------------------------------------
# Initializers shadowing graph inputs — explicit policy: bake in, but warn
# ---------------------------------------------------------------------------


def _shadowed_input_model():
    return _model(
        [helper.make_node("Add", ["x", "c"], ["y"])],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        initializers=[
            numpy_helper.from_array(np.array([5.0, 5.0], dtype=np.float32), "x"),
            numpy_helper.from_array(np.array([1.0, 1.0], dtype=np.float32), "c"),
        ],
    )


def test_fold_constants_warns_on_initializer_shadowing_input():
    # Per the ONNX IR spec the initializer is a caller-overridable default for
    # the input; folding freezes it, so the caller must at least be told.
    m = _shadowed_input_model()
    with pytest.warns(UserWarning, match="shadow"):
        m = fold_constants(m)
    assert not m.graph.node  # policy unchanged: shadowed inputs are baked in
    assert {i.name for i in m.graph.initializer} >= {"y"}


def test_fold_shadow_warning_stacklevel():
    """When called via preprocess(), the shadowing warning must point at
    preprocess()'s caller (this test file), not inside the _passes package
    (same contract as test_normalize_opset_warning_stacklevel)."""
    m = _shadowed_input_model()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preprocess(m)
    shadow = [x for x in w if "shadow graph inputs" in str(x.message)]
    assert len(shadow) == 1
    filename = shadow[0].filename
    assert "coreai_onnx/_passes" not in filename.replace("\\", "/"), (
        f"Warning points at {filename!r}; expected caller's file, not the internal module"
    )


def test_preprocess_keeps_initializer_shadowing_graph_input():
    # The shadowing initializer defines the converted signature (the input is
    # dropped), so dead-initializer pruning must keep it even when no node
    # consumes it after folding.
    m = _shadowed_input_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the shadowing warning, tested above
        m = preprocess(m)
    assert "x" in {i.name for i in m.graph.initializer}


# ---------------------------------------------------------------------------
# Dead-initializer pruning
# ---------------------------------------------------------------------------


def test_preprocess_prunes_dead_initializers():
    # Cast(w_f32 -> f16) folds; the f32 source must not survive (the converter
    # emits a coreai.constant per initializer, doubling the weight memory).
    m = _model(
        [
            helper.make_node("Cast", ["w"], ["w16"], to=TensorProto.FLOAT16),
            helper.make_node("MatMul", ["x", "w16"], ["y"]),
        ],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [3, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [3, 4])],
        initializers=[numpy_helper.from_array(np.ones((8, 4), dtype=np.float32), "w")],
    )
    m = preprocess(m)
    names = {i.name for i in m.graph.initializer}
    assert "w16" in names
    assert "w" not in names


# ---------------------------------------------------------------------------
# Recursive preprocessing inside control-flow subgraphs
# ---------------------------------------------------------------------------


def _branch_attr(if_node, name):
    return next(a.g for a in if_node.attribute if a.name == name)


def test_preprocess_cleans_if_branch_nodes_recursively():
    then_g = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["t"]),
            helper.make_node("Relu", ["t"], ["then_out"]),
            helper.make_node("Sigmoid", ["x"], ["dead"]),
        ],
        "then_g",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_g = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["e"]),
            helper.make_node("Neg", ["e"], ["else_out"]),
        ],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    model = _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )

    model = preprocess(model)
    (if_node,) = model.graph.node
    then_g = _branch_attr(if_node, "then_branch")
    else_g = _branch_attr(if_node, "else_branch")
    assert [n.op_type for n in then_g.node] == ["Relu"]
    assert list(then_g.node[0].input) == ["x"]
    assert [n.op_type for n in else_g.node] == ["Neg"]
    assert list(else_g.node[0].input) == ["x"]


def test_preprocess_folds_and_prunes_if_branch_initializers_recursively():
    then_g = helper.make_graph(
        [
            helper.make_node("Add", ["c1", "c2"], ["bias"]),
            helper.make_node("Add", ["x", "bias"], ["then_out"]),
        ],
        "then_g",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
        initializer=[
            numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "c1"),
            numpy_helper.from_array(np.array([3.0, 4.0], dtype=np.float32), "c2"),
            numpy_helper.from_array(np.array([9.0, 9.0], dtype=np.float32), "unused"),
        ],
    )
    else_g = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["else_out"])],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    model = _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )

    model = preprocess(model)
    (if_node,) = model.graph.node
    then_g = _branch_attr(if_node, "then_branch")
    assert [n.op_type for n in then_g.node] == ["Add"]
    init_names = {i.name for i in then_g.initializer}
    assert init_names == {"bias"}
    (bias,) = [i for i in then_g.initializer if i.name == "bias"]
    np.testing.assert_array_equal(numpy_helper.to_array(bias), [4.0, 6.0])


@coreai_test
@pytest.mark.ir
def test_converter_pipeline_uses_recursive_branch_cleanup():
    import coreai_onnx

    then_g = helper.make_graph(
        [
            helper.make_node("Identity", ["x"], ["t"]),
            helper.make_node("Relu", ["t"], ["then_out"]),
            helper.make_node("Sigmoid", ["x"], ["dead"]),
        ],
        "then_g",
        [],
        [helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [2])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["else_out"])],
        "else_g",
        [],
        [helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [2])],
    )
    model = _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )

    text = str(coreai_onnx.convert(model))
    assert "coreai.sigmoid" not in text
    assert "coreai.copy" not in text


# ---------------------------------------------------------------------------
# GroupNormalization opset 18-20 (deprecated per-group form)
# ---------------------------------------------------------------------------


def _gn18_model(opset=18, sym_channels=False):
    """Relu -> GroupNormalization with num_groups=3 != C=6, per-group params."""
    c_dim = "C" if sym_channels else 6
    g = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["r"]),
            helper.make_node(
                "GroupNormalization",
                ["r", "scale", "bias"],
                ["y"],
                num_groups=3,
                epsilon=1e-5,
            ),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, c_dim, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, c_dim, 4, 4])],
        initializer=[
            numpy_helper.from_array(np.array([1.5, 2.0, 0.5], np.float32), "scale"),
            numpy_helper.from_array(np.array([0.1, -0.2, 0.3], np.float32), "bias"),
        ],
    )
    return helper.make_model(
        g, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )


@pytest.mark.parametrize("opset", [18, 19, 20])
def test_preprocess_upgrades_deprecated_group_normalization(opset):
    # The checker rejects GroupNormalization at opset 18-20 ("deprecated in
    # domain_version of 18"); preprocess must upgrade to the opset-21
    # per-channel form, expanding per-group scale/bias.
    m = _gn18_model(opset)
    x = np.random.default_rng(0).standard_normal((2, 6, 4, 4)).astype(np.float32)
    expected = run_onnxruntime(m, {"x": x})
    m = preprocess(m)
    (dom,) = [o for o in m.opset_import if o.domain == ""]
    assert dom.version >= 21
    gn = next(n for n in m.graph.node if n.op_type == "GroupNormalization")
    inits = {i.name: i for i in m.graph.initializer}
    assert list(inits[gn.input[1]].dims) == [6]
    assert list(inits[gn.input[2]].dims) == [6]
    got = run_onnxruntime(m, {"x": x})
    np.testing.assert_allclose(got[0], expected[0], rtol=1e-5, atol=1e-6)


def test_preprocess_rejects_group_normalization_with_unknown_channels():
    # C cannot be resolved statically: the model is left as-is and the checker
    # rejects it (the per-group semantics must never reach the per-channel
    # lowering silently).
    m = _gn18_model(sym_channels=True)
    with pytest.raises(onnx.checker.ValidationError):
        preprocess(m)


def test_preprocess_rejects_group_normalization_with_mismatched_param_shape():
    # num_groups=2 but per-group scale/bias have length 3 (invalid opset-18 GN).
    # 6 % 3 == 0 slips past a divisibility-only guard; preprocess must leave the
    # model untouched so the checker rejects it, not expand bogus per-channel
    # params into a valid-looking opset-21 model.
    g = helper.make_graph(
        [
            helper.make_node("Relu", ["x"], ["r"]),
            helper.make_node(
                "GroupNormalization",
                ["r", "scale", "bias"],
                ["y"],
                num_groups=2,
                epsilon=1e-5,
            ),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 6, 4, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6, 4, 4])],
        initializer=[
            numpy_helper.from_array(np.array([1.5, 2.0, 0.5], np.float32), "scale"),
            numpy_helper.from_array(np.array([0.1, -0.2, 0.3], np.float32), "bias"),
        ],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)
    with pytest.raises(onnx.checker.ValidationError):
        preprocess(m)


@coreai_runtime_test
async def test_parity_group_normalization_opset18():
    m = _gn18_model()
    x = np.random.default_rng(1).standard_normal((2, 6, 4, 4)).astype(np.float32)
    await assert_parity(m, {"x": x})


# ---------------------------------------------------------------------------
# >2 GiB models — check_model/infer_shapes must not serialize weight payloads
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_preprocess_handles_over_2gib_model():
    """A >2 GiB proto used to crash with protobuf EncodeError: check_model and
    infer_shapes serialize their whole argument."""
    n = 290_000_000  # 2 x 1.16 GB fp32 puts the proto over the 2 GiB limit
    g = helper.make_graph(
        [
            helper.make_node("Mul", ["w0", "w1"], ["wide"]),
            helper.make_node("Add", ["x", "c"], ["s"]),
            helper.make_node("Softmax", ["s"], ["y"], axis=-1),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("wide", TensorProto.FLOAT, [n]),
        ],
        initializer=[numpy_helper.from_array(np.ones(4, dtype=np.float32), "c")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 22)], ir_version=10)
    for name in ("w0", "w1"):
        t = m.graph.initializer.add()
        t.name = name
        t.data_type = TensorProto.FLOAT
        t.dims.append(n)
        t.raw_data = b"\x00" * (4 * n)

    m = preprocess(m)  # raised google.protobuf.message.EncodeError before
    m = fuse_attention(m)  # second infer_shapes call site, same ceiling
    assert {i.name for i in m.graph.initializer} >= {"w0", "w1"}
    assert [n.op_type for n in m.graph.node] == ["Mul", "Add", "Softmax"]
