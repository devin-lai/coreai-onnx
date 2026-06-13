# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for GELU/SiLU chain fusion (coreai_onnx._fusion.fuse_activations).

torch.onnx.export at the opset-17 baseline emits nn.GELU as a 5-node erf chain
(9 nodes for the tanh approximation) and SiLU/Swish as Sigmoid->Mul; without
fusion each becomes that many full-tensor elementwise kernels even though the
coreai dialect ships dedicated gelu/silu primitives.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import coreai_onnx
from coreai_onnx._fusion import fuse_activations
from coreai_onnx._utils import attrs as _node_attrs

from .helpers import assert_parity, coreai_runtime_test, coreai_test

_SQRT2 = 1.4142135381698608  # f32-rounded constants as torch bakes them
_RSQRT2 = 0.7071067690849304
_BETA = 0.7978845834732056  # sqrt(2/pi)
_CUBE = 0.044715


def _vi(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _scalar(name, v):
    return numpy_helper.from_array(np.array(v, dtype=np.float32), name)


def _model(nodes, inputs, outputs, initializers=(), opset=17):
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializers)
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10
    )


def _gelu_erf_model(B=2, D=8):
    """torch opset-17 layout: Div -> Erf -> Add(1) -> Mul(x) -> Mul(0.5)."""
    return _model(
        [
            helper.make_node("Div", ["x", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["a"]),
            helper.make_node("Mul", ["x", "a"], ["m"]),
            helper.make_node("Mul", ["m", "half"], ["y"]),
        ],
        [_vi("x", [B, D])],
        [_vi("y", [B, D])],
        [_scalar("sqrt2", _SQRT2), _scalar("one", 1.0), _scalar("half", 0.5)],
    )


def _gelu_tanh_model(B=2, D=8, cube_via_pow=False):
    """torch opset-17 layout: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))."""
    if cube_via_pow:
        cube_nodes = [helper.make_node("Pow", ["x", "three"], ["x3"])]
        cube_inits = [_scalar("three", 3.0)]
    else:
        cube_nodes = [
            helper.make_node("Mul", ["x", "x"], ["xx"]),
            helper.make_node("Mul", ["x", "xx"], ["x3"]),
        ]
        cube_inits = []
    return _model(
        [
            *cube_nodes,
            helper.make_node("Mul", ["cube", "x3"], ["cx3"]),
            helper.make_node("Add", ["x", "cx3"], ["inner"]),
            helper.make_node("Mul", ["beta", "inner"], ["bi"]),
            helper.make_node("Tanh", ["bi"], ["t"]),
            helper.make_node("Add", ["one", "t"], ["a"]),
            helper.make_node("Mul", ["x", "a"], ["m"]),
            helper.make_node("Mul", ["half", "m"], ["y"]),
        ],
        [_vi("x", [B, D])],
        [_vi("y", [B, D])],
        [
            *cube_inits,
            _scalar("cube", _CUBE),
            _scalar("beta", _BETA),
            _scalar("one", 1.0),
            _scalar("half", 0.5),
        ],
    )


def _silu_model(B=2, D=8, swapped=False):
    mul_inputs = ["x", "s"] if swapped else ["s", "x"]
    return _model(
        [
            helper.make_node("Sigmoid", ["x"], ["s"]),
            helper.make_node("Mul", mul_inputs, ["y"]),
        ],
        [_vi("x", [B, D])],
        [_vi("y", [B, D])],
    )


def _if_silu_model(B=2, D=8):
    then_g = helper.make_graph(
        [
            helper.make_node("Sigmoid", ["x"], ["s"]),
            helper.make_node("Mul", ["s", "x"], ["then_y"]),
        ],
        "then",
        [],
        [_vi("then_y", [B, D])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["else_y"])],
        "else",
        [],
        [_vi("else_y", [B, D])],
    )
    return _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            _vi("x", [B, D]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("y", [B, D])],
    )


def _if_gelu_erf_model(B=2, D=8):
    then_g = helper.make_graph(
        [
            helper.make_node("Div", ["x", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["a"]),
            helper.make_node("Mul", ["x", "a"], ["m"]),
            helper.make_node("Mul", ["m", "half"], ["then_y"]),
        ],
        "then",
        [],
        [_vi("then_y", [B, D])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["else_y"])],
        "else",
        [],
        [_vi("else_y", [B, D])],
    )
    return _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            _vi("x", [B, D]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("y", [B, D])],
        [_scalar("sqrt2", _SQRT2), _scalar("one", 1.0), _scalar("half", 0.5)],
    )


def _if_gelu_erf_shadowed_model(B=2, D=8):
    """As _if_gelu_erf_model, but the then-branch divides by a LOCAL, non-scalar
    ``sqrt2`` (a real per-element divisor) that shadows the outer scalar of the
    same name. The chain must NOT fuse: the local ``sqrt2`` is not GELU's
    1/sqrt(2) constant, so collapsing it to Gelu would silently change results."""
    then_g = helper.make_graph(
        [
            helper.make_node("Div", ["x", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["a"]),
            helper.make_node("Mul", ["x", "a"], ["m"]),
            helper.make_node("Mul", ["m", "half"], ["then_y"]),
        ],
        "then",
        [],
        [_vi("then_y", [B, D])],
        initializer=[
            numpy_helper.from_array(np.full((B, D), _SQRT2, dtype=np.float32), "sqrt2")
        ],
    )
    else_g = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["else_y"])],
        "else",
        [],
        [_vi("else_y", [B, D])],
    )
    return _model(
        [
            helper.make_node(
                "If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            _vi("x", [B, D]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("y", [B, D])],
        [_scalar("sqrt2", _SQRT2), _scalar("one", 1.0), _scalar("half", 0.5)],
    )


def _then_branch(model):
    (if_node,) = list(model.graph.node)
    return next(a.g for a in if_node.attribute if a.name == "then_branch")


def _op_types(model):
    return [n.op_type for n in model.graph.node]


# ---------------------------------------------------------------------------
# GELU — positive cases
# ---------------------------------------------------------------------------


def test_fuse_gelu_erf_torch_layout():
    fused = fuse_activations(_gelu_erf_model())
    (gelu,) = [n for n in fused.graph.node if n.op_type == "Gelu"]
    assert _op_types(fused) == ["Gelu"]
    assert list(gelu.input) == ["x"]
    assert gelu.output[0] == "y"
    assert _node_attrs(gelu)["approximate"] == "none"


def test_fuse_purges_stale_value_info():
    # In the conversion pipeline fuse_activations runs after preprocess, whose
    # shape inference populates value_info for every intermediate; entries for
    # the deleted chain nodes must not survive in the fused output.
    model = onnx.shape_inference.infer_shapes(_gelu_erf_model())
    assert any(vi.name == "e" for vi in model.graph.value_info)
    fused = fuse_activations(model)
    assert _op_types(fused) == ["Gelu"]
    produced = {out for n in fused.graph.node for out in n.output}
    known = (
        produced
        | {vi.name for vi in fused.graph.input}
        | {init.name for init in fused.graph.initializer}
    )
    stale = [vi.name for vi in fused.graph.value_info if vi.name not in known]
    assert stale == []


def test_fuse_gelu_erf_mul_by_rsqrt2():
    # Some exporters emit Mul(x, 1/sqrt(2)) instead of Div(x, sqrt(2)).
    model = _gelu_erf_model()
    (div,) = [n for n in model.graph.node if n.op_type == "Div"]
    div.op_type = "Mul"
    for init in model.graph.initializer:
        if init.name == "sqrt2":
            init.CopyFrom(_scalar("sqrt2", _RSQRT2))
    fused = fuse_activations(model)
    assert _op_types(fused) == ["Gelu"]


def test_fuse_gelu_erf_half_x_tree():
    # Alternative association: y = (x * 0.5) * (1 + erf(x / sqrt(2))).
    model = _model(
        [
            helper.make_node("Div", ["x", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["a"]),
            helper.make_node("Mul", ["x", "half"], ["hx"]),
            helper.make_node("Mul", ["hx", "a"], ["y"]),
        ],
        [_vi("x", [2, 8])],
        [_vi("y", [2, 8])],
        [_scalar("sqrt2", _SQRT2), _scalar("one", 1.0), _scalar("half", 0.5)],
    )
    fused = fuse_activations(model)
    assert _op_types(fused) == ["Gelu"]


def test_fuse_gelu_tanh_torch_layout():
    fused = fuse_activations(_gelu_tanh_model())
    (gelu,) = [n for n in fused.graph.node if n.op_type == "Gelu"]
    assert _op_types(fused) == ["Gelu"]
    assert list(gelu.input) == ["x"]
    assert _node_attrs(gelu)["approximate"] == "tanh"


def test_fuse_gelu_tanh_pow_cube():
    fused = fuse_activations(_gelu_tanh_model(cube_via_pow=True))
    assert _op_types(fused) == ["Gelu"]
    (gelu,) = list(fused.graph.node)
    assert _node_attrs(gelu)["approximate"] == "tanh"


# ---------------------------------------------------------------------------
# GELU — negative cases (must stay untouched)
# ---------------------------------------------------------------------------


def _assert_not_fused(model):
    before = _op_types(model)
    fused = fuse_activations(model)
    assert _op_types(fused) == before


def test_no_fuse_gelu_wrong_divisor():
    model = _gelu_erf_model()
    for init in model.graph.initializer:
        if init.name == "sqrt2":
            init.CopyFrom(_scalar("sqrt2", 1.5))
    _assert_not_fused(model)


def test_no_fuse_gelu_wrong_half():
    model = _gelu_erf_model()
    for init in model.graph.initializer:
        if init.name == "half":
            init.CopyFrom(_scalar("half", 0.6))
    _assert_not_fused(model)


def test_no_fuse_gelu_intermediate_is_graph_output():
    model = _gelu_erf_model()
    model.graph.output.append(_vi("e", [2, 8]))
    _assert_not_fused(model)


def test_no_fuse_gelu_intermediate_multi_consumer():
    model = _gelu_erf_model()
    model.graph.node.append(helper.make_node("Relu", ["a"], ["a_relu"]))
    model.graph.output.append(_vi("a_relu", [2, 8]))
    _assert_not_fused(model)


def test_no_fuse_gelu_mismatched_x():
    # The outer Mul multiplies a different tensor than the Erf input.
    model = _model(
        [
            helper.make_node("Div", ["x", "sqrt2"], ["d"]),
            helper.make_node("Erf", ["d"], ["e"]),
            helper.make_node("Add", ["e", "one"], ["a"]),
            helper.make_node("Mul", ["x2", "a"], ["m"]),
            helper.make_node("Mul", ["m", "half"], ["y"]),
        ],
        [_vi("x", [2, 8]), _vi("x2", [2, 8])],
        [_vi("y", [2, 8])],
        [_scalar("sqrt2", _SQRT2), _scalar("one", 1.0), _scalar("half", 0.5)],
    )
    _assert_not_fused(model)


# ---------------------------------------------------------------------------
# SiLU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("swapped", [False, True])
def test_fuse_silu(swapped):
    fused = fuse_activations(_silu_model(swapped=swapped))
    (silu,) = list(fused.graph.node)
    assert silu.op_type == "Silu"
    assert silu.domain == "coreai_onnx"
    assert list(silu.input) == ["x"]
    assert silu.output[0] == "y"
    assert any(oi.domain == "coreai_onnx" for oi in fused.opset_import)


def test_no_fuse_silu_mul_with_other_tensor():
    model = _model(
        [
            helper.make_node("Sigmoid", ["x"], ["s"]),
            helper.make_node("Mul", ["s", "z"], ["y"]),
        ],
        [_vi("x", [2, 8]), _vi("z", [2, 8])],
        [_vi("y", [2, 8])],
    )
    _assert_not_fused(model)


def test_no_fuse_silu_sigmoid_multi_consumer():
    model = _silu_model()
    model.graph.node.append(helper.make_node("Relu", ["s"], ["s_relu"]))
    model.graph.output.append(_vi("s_relu", [2, 8]))
    _assert_not_fused(model)


def test_no_fuse_silu_sigmoid_is_graph_output():
    model = _silu_model()
    model.graph.output.append(_vi("s", [2, 8]))
    _assert_not_fused(model)


# ---------------------------------------------------------------------------
# Control-flow subgraphs
# ---------------------------------------------------------------------------


def test_fuse_silu_inside_if_branch():
    fused = fuse_activations(_if_silu_model())
    then_g = _then_branch(fused)
    (silu,) = list(then_g.node)
    assert silu.op_type == "Silu"
    assert silu.domain == "coreai_onnx"
    assert any(oi.domain == "coreai_onnx" for oi in fused.opset_import)


def test_fuse_gelu_inside_if_branch_with_outer_scalars():
    fused = fuse_activations(_if_gelu_erf_model())
    then_g = _then_branch(fused)
    (gelu,) = list(then_g.node)
    assert gelu.op_type == "Gelu"
    assert list(gelu.input) == ["x"]
    assert _node_attrs(gelu)["approximate"] == "none"


def test_no_fuse_when_local_binding_shadows_outer_scalar_in_branch():
    # A subgraph name bound locally (here a non-scalar initializer; likewise a
    # Loop/Scan formal input) must hide an outer scalar of the same name, so the
    # stale outer constant cannot drive a wrong fusion.
    fused = fuse_activations(_if_gelu_erf_shadowed_model())
    then_g = _then_branch(fused)
    assert [n.op_type for n in then_g.node] == ["Div", "Erf", "Add", "Mul", "Mul"]


# ---------------------------------------------------------------------------
# Emitted IR (conversion only, no runtime) — the converter pipeline must fuse
# ---------------------------------------------------------------------------


@coreai_test
@pytest.mark.ir
def test_silu_lowered_to_coreai_silu():
    text = str(coreai_onnx.convert(_silu_model()))
    assert "coreai.silu" in text
    assert "sigmoid" not in text


@coreai_test
@pytest.mark.ir
def test_gelu_erf_lowered_to_coreai_gelu():
    text = str(coreai_onnx.convert(_gelu_erf_model()))
    assert "coreai.gelu" in text
    assert "erf" not in text


@coreai_test
@pytest.mark.ir
def test_gelu_tanh_lowered_to_coreai_gelu():
    text = str(coreai_onnx.convert(_gelu_tanh_model()))
    assert "coreai.gelu" in text
    assert "coreai.tanh" not in text  # only the approximate attr mentions tanh


@coreai_test
@pytest.mark.ir
def test_silu_inside_if_branch_lowered_to_coreai_silu():
    text = str(coreai_onnx.convert(_if_silu_model()))
    assert "coreai.silu" in text
    assert "coreai.sigmoid" not in text


@coreai_test
@pytest.mark.ir
def test_gelu_inside_if_branch_lowered_to_coreai_gelu():
    text = str(coreai_onnx.convert(_if_gelu_erf_model()))
    assert "coreai.gelu" in text
    assert "coreai.erf" not in text


# ---------------------------------------------------------------------------
# Numerical parity (Core AI runtime) — ORT runs the unfused chain
# ---------------------------------------------------------------------------


def _feeds(seed=0):
    rng = np.random.default_rng(seed)
    return {"x": rng.standard_normal((2, 8)).astype(np.float32)}


@coreai_runtime_test
async def test_parity_gelu_erf():
    await assert_parity(_gelu_erf_model(), _feeds())


@coreai_runtime_test
async def test_parity_gelu_tanh():
    await assert_parity(_gelu_tanh_model(), _feeds(1))


@coreai_runtime_test
async def test_parity_silu():
    await assert_parity(_silu_model(), _feeds(2))
