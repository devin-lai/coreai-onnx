# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for attention fusion (coreai_onnx._fusion) and its lowering.

The fused composite is what keeps the on-device GPU compiler from crashing on
raw attention chains (MPSGraph: "MLIR pass manager failed"), so these tests
cover three layers: the ONNX-level pattern rewrite, the emitted IR, and
numerical parity of converted models — including the GPU/Neural-Engine
regression where transpose-fed composite operands silently mis-execute.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import coreai_onnx
from coreai_onnx._fusion import fuse_attention
from coreai_onnx._utils import attrs as _node_attrs
from coreai_onnx.errors import ConversionError

from .helpers import assert_parity, requires_coreai_runtime, run_aimodel

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _vi(name: str, shape) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _model(nodes, inputs, outputs, initializers=(), with_coreai_opset=False):
    opsets = [helper.make_opsetid("", 22)]
    if with_coreai_opset:
        opsets.append(helper.make_opsetid("coreai", 1))
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializers)
    )
    return helper.make_model(graph, opset_imports=opsets, ir_version=10)


def _yolo_model(B=1, H=2, L=9, dk=4, dv=8, scale=0.5):
    """Ultralytics C2PSA shape: out = v @ softmax((q @ kt) * scale)^T."""
    return _model(
        [
            helper.make_node("MatMul", ["q", "kt"], ["s"]),
            helper.make_node("Mul", ["s", "c"], ["sc"]),
            helper.make_node("Softmax", ["sc"], ["p"], axis=3),
            helper.make_node("Transpose", ["p"], ["pt"], perm=[0, 1, 3, 2]),
            helper.make_node("MatMul", ["v", "pt"], ["out"]),
        ],
        [_vi("q", [B, H, L, dk]), _vi("kt", [B, H, dk, L]), _vi("v", [B, H, dv, L])],
        [_vi("out", [B, H, dv, L])],
        [numpy_helper.from_array(np.array(scale, dtype=np.float32), name="c")],
    )


def _standard_model(
    B=2,
    H=3,
    L=5,
    S=7,
    E=8,
    *,
    scale_op="Div",
    scale=2.5,
    mask=False,
    rank=4,
    dk=None,
    dv=None,
):
    """Transformer shape: out = softmax(q @ kt [/ or *] scale [+ mask]) @ v.

    ``dk``/``dv`` default to ``E``; set them apart for the asymmetric head dims
    (e.g. Conditional DETR cross-attention uses key_dim > value_dim).
    """
    dk = E if dk is None else dk
    dv = E if dv is None else dv
    lead = [B, H] if rank == 4 else [B]
    nodes = [helper.make_node("MatMul", ["q", "kt"], ["s"])]
    probs_in = "s"
    if scale_op is not None:
        nodes.append(helper.make_node(scale_op, ["s", "c"], ["sc"]))
        probs_in = "sc"
    if mask:
        nodes.append(helper.make_node("Add", [probs_in, "mask"], ["sm"]))
        probs_in = "sm"
    nodes += [
        helper.make_node("Softmax", [probs_in], ["p"], axis=-1),
        helper.make_node("MatMul", ["p", "v"], ["out"]),
    ]
    inputs = [
        _vi("q", [*lead, L, dk]),
        _vi("kt", [*lead, dk, S]),
        _vi("v", [*lead, S, dv]),
    ]
    if mask:
        inputs.append(_vi("mask", [L, S]))
    initializers = (
        [numpy_helper.from_array(np.array(scale, dtype=np.float32), name="c")]
        if scale_op is not None
        else []
    )
    return _model(nodes, inputs, [_vi("out", [*lead, L, dv])], initializers)


def _prescaled_model(B=1, H=4, L=16, E=16, s=0.25):
    """torch MultiheadAttention-style export: q and k pre-scaled by a runtime
    Sqrt, no scale op between MatMul and Softmax (fuses with scale == 1)."""
    return _model(
        [
            helper.make_node("Sqrt", ["s0"], ["sq1"]),
            helper.make_node("Sqrt", ["s0"], ["sq2"]),
            helper.make_node("Mul", ["qin", "sq1"], ["q"]),
            helper.make_node("Transpose", ["kin"], ["kt0"], perm=[0, 1, 3, 2]),
            helper.make_node("Mul", ["kt0", "sq2"], ["kt"]),
            helper.make_node("MatMul", ["q", "kt"], ["sc"]),
            helper.make_node("Softmax", ["sc"], ["p"], axis=-1),
            helper.make_node("MatMul", ["p", "vin"], ["out"]),
        ],
        [_vi("qin", [B, H, L, E]), _vi("kin", [B, H, L, E]), _vi("vin", [B, H, L, E])],
        [_vi("out", [B, H, L, E])],
        [numpy_helper.from_array(np.array(s, dtype=np.float32), name="s0")],
    )


def _sdpa_nodes(model):
    return [n for n in model.graph.node if n.op_type == "ScaledDotProductAttention"]


def _op_types(model):
    return [n.op_type for n in model.graph.node]


def _random_feeds(model, seed=0):
    rng = np.random.default_rng(seed)
    return {
        vi.name: rng.standard_normal(
            [d.dim_value for d in vi.type.tensor_type.shape.dim]
        ).astype(np.float32)
        for vi in model.graph.input
    }


def _reference_sdpa(q, k, v, scale, mask=None):
    s = (q.astype(np.float64) * scale) @ k.astype(np.float64).swapaxes(-1, -2)
    if mask is not None:
        s = s + mask
    e = np.exp(s - s.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)
    return (p @ v.astype(np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion pass — positive cases
# ---------------------------------------------------------------------------


def test_fuse_yolo_orientation():
    fused = fuse_attention(_yolo_model())
    (sdpa,) = _sdpa_nodes(fused)
    assert sdpa.domain == "coreai"
    assert len(sdpa.input) == 3
    assert _node_attrs(sdpa)["scale"] == pytest.approx(0.5)
    # K transpose, V transpose, output transpose; q and k zero-padded 4 -> 8.
    ops = _op_types(fused)
    assert ops.count("Transpose") == 3
    assert ops.count("Pad") == 2
    assert not (set(ops) & {"MatMul", "Softmax", "Mul"})
    assert any(oi.domain == "coreai" for oi in fused.opset_import)


def test_fuse_standard_orientation_div_scale():
    fused = fuse_attention(_standard_model())
    (sdpa,) = _sdpa_nodes(fused)
    assert _node_attrs(sdpa)["scale"] == pytest.approx(1 / 2.5)
    # Equal head dims: only the K-canonicalizing transpose is added.
    assert _op_types(fused) == ["Transpose", "ScaledDotProductAttention"]


def test_fuse_with_additive_mask():
    fused = fuse_attention(_standard_model(mask=True))
    (sdpa,) = _sdpa_nodes(fused)
    assert list(sdpa.input)[3] == "mask"


def test_fuse_rank3():
    fused = fuse_attention(_standard_model(rank=3))
    assert len(_sdpa_nodes(fused)) == 1


def test_fuse_without_scale_op_uses_scale_one():
    fused = fuse_attention(_standard_model(scale_op=None))
    (sdpa,) = _sdpa_nodes(fused)
    assert _node_attrs(sdpa)["scale"] == pytest.approx(1.0)


def test_fuse_mul_scale_commuted():
    model = _standard_model(scale_op="Mul")
    (mul,) = [n for n in model.graph.node if n.op_type == "Mul"]
    mul.input[0], mul.input[1] = mul.input[1], mul.input[0]
    fused = fuse_attention(model)
    (sdpa,) = _sdpa_nodes(fused)
    assert _node_attrs(sdpa)["scale"] == pytest.approx(2.5)


def test_fuse_keeps_surrounding_nodes():
    model = _standard_model()
    g = model.graph
    g.node.insert(0, helper.make_node("Relu", ["q"], ["q_relu"]))
    g.node[1].input[0] = "q_relu"  # scores MatMul consumes the Relu
    fused = fuse_attention(model)
    assert _op_types(fused) == ["Relu", "Transpose", "ScaledDotProductAttention"]
    assert _sdpa_nodes(fused)[0].input[0] == "q_relu"


def test_fuse_purges_stale_value_info():
    # fuse_attention runs shape inference (populating value_info for every
    # intermediate) and then deletes the matched chain nodes; value_info
    # entries for the deleted intermediates must not survive in the output.
    fused = fuse_attention(_standard_model())
    assert _sdpa_nodes(fused)  # the chain did fuse
    produced = {out for n in fused.graph.node for out in n.output}
    known = (
        produced
        | {vi.name for vi in fused.graph.input}
        | {init.name for init in fused.graph.initializer}
    )
    stale = [vi.name for vi in fused.graph.value_info if vi.name not in known]
    assert stale == []


# ---------------------------------------------------------------------------
# Fusion pass — negative cases (must stay untouched)
# ---------------------------------------------------------------------------


def _assert_not_fused(model):
    before = _op_types(model)
    fused = fuse_attention(model)
    assert _op_types(fused) == before
    assert not _sdpa_nodes(fused)


def test_no_fuse_softmax_not_last_axis():
    model = _standard_model()
    (softmax,) = [n for n in model.graph.node if n.op_type == "Softmax"]
    softmax.attribute[0].i = 2
    _assert_not_fused(model)


def test_no_fuse_probs_is_graph_output():
    model = _standard_model()
    model.graph.output.append(_vi("p", [2, 3, 5, 7]))
    _assert_not_fused(model)


def test_no_fuse_probs_has_second_consumer():
    model = _standard_model()
    model.graph.node.append(helper.make_node("Relu", ["p"], ["p2"]))
    model.graph.output.append(_vi("p2", [2, 3, 5, 7]))
    _assert_not_fused(model)


def test_no_fuse_non_scalar_scale():
    model = _standard_model()
    del model.graph.initializer[:]
    model.graph.initializer.append(
        numpy_helper.from_array(np.full((7,), 2.5, dtype=np.float32), name="c")
    )
    _assert_not_fused(model)


def test_fuse_query_dim_larger_than_value_dim_standard():
    # Conditional DETR cross-attention: dk(8) > dv(4). value is zero-padded up
    # to dk and the fused output sliced back to dv.
    fused = fuse_attention(_standard_model(dk=8, dv=4))
    (sdpa,) = _sdpa_nodes(fused)
    assert sdpa.domain == "coreai"
    ops = _op_types(fused)
    assert ops.count("Pad") == 1  # value 4 -> 8
    assert ops.count("Slice") == 1  # output 8 -> 4
    assert not (set(ops) & {"MatMul", "Softmax", "Div"})


def test_fuse_query_dim_larger_than_value_dim_transposed():
    # The same widening works in the YOLO (transposed-output) orientation.
    fused = fuse_attention(_yolo_model(dk=8, dv=4))
    assert len(_sdpa_nodes(fused)) == 1
    ops = _op_types(fused)
    assert ops.count("Pad") == 1
    assert ops.count("Slice") == 1
    assert not (set(ops) & {"MatMul", "Softmax", "Mul"})


def test_no_fuse_probs_as_second_matmul_operand():
    # out = x @ probs contracts over the query axis — not attention.
    model = _standard_model(L=7)  # square L == S so the swap stays valid
    (out_mm,) = [n for n in model.graph.node if n.output[0] == "out"]
    out_mm.input[0], out_mm.input[1] = "v", "p"
    (v_in,) = [i for i in model.graph.input if i.name == "v"]
    del v_in.type.tensor_type.shape.dim[:]
    for d in [2, 3, 8, 7]:
        v_in.type.tensor_type.shape.dim.add().dim_value = d
    (out_vi,) = list(model.graph.output)
    del out_vi.type.tensor_type.shape.dim[:]
    for d in [2, 3, 8, 7]:
        out_vi.type.tensor_type.shape.dim.add().dim_value = d
    _assert_not_fused(model)


def test_no_fuse_dynamic_shapes():
    model = _standard_model()
    for name in ("q", "kt"):
        (vi,) = [i for i in model.graph.input if i.name == name]
        dim = vi.type.tensor_type.shape.dim[2 if name == "q" else 3]
        dim.ClearField("dim_value")
        dim.dim_param = "seq"
    _assert_not_fused(model)


def test_no_fuse_wrong_probs_transpose_perm():
    model = _yolo_model()  # L == S, so an identity perm keeps shapes valid
    (transpose,) = [n for n in model.graph.node if n.op_type == "Transpose"]
    transpose.attribute[0].ints[:] = [0, 1, 2, 3]
    _assert_not_fused(model)


def test_no_fuse_non_finite_scale():
    # 1/denormal overflows float32; baking an inf scale into the composite
    # would corrupt the kernel output, so the chain must stay unfused.
    model = _standard_model(scale=1e-40)
    _assert_not_fused(model)


def test_no_fuse_probs_consumed_only_in_subgraph():
    # The softmax output is captured by exactly one If branch: a single
    # consumer overall, but not a top-level node input.  Regression: this
    # used to raise KeyError instead of skipping the chain.
    B, H, L, S, E = 1, 2, 4, 4, 8
    then_g = helper.make_graph(
        [helper.make_node("Relu", ["p"], ["r"])], "then", [], [_vi("r", [B, H, L, S])]
    )
    else_g = helper.make_graph(
        [
            helper.make_node(
                "ConstantOfShape",
                ["shp"],
                ["r2"],
                value=numpy_helper.from_array(np.array([0.0], dtype=np.float32)),
            )
        ],
        "else",
        [],
        [_vi("r2", None)],
    )
    model = _model(
        [
            helper.make_node("MatMul", ["q", "kt"], ["s"]),
            helper.make_node("Softmax", ["s"], ["p"], axis=-1),
            helper.make_node(
                "If", ["cond"], ["out"], then_branch=then_g, else_branch=else_g
            ),
        ],
        [
            _vi("q", [B, H, L, E]),
            _vi("kt", [B, H, E, S]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("out", [B, H, L, S])],
        [numpy_helper.from_array(np.array([B, H, L, S], dtype=np.int64), name="shp")],
    )
    _assert_not_fused(model)


def _assert_no_dangling_inputs(model):
    produced = (
        {o for n in model.graph.node for o in n.output}
        | {i.name for i in model.graph.initializer}
        | {i.name for i in model.graph.input}
    )
    dangling = [
        (n.op_type, i) for n in model.graph.node for i in n.input if i not in produced
    ]
    assert not dangling, f"dangling node inputs after fusion: {dangling}"


def test_overlapping_matches_fuse_only_first():
    # Stacked attention where the first chain's output MatMul is the second
    # chain's scores MatMul.  Both candidates match individually; fusing both
    # would delete the first softmax while the second fused node still
    # references its output.  Regression: only the first (graph-order) chain
    # may fuse, and the result must stay well-formed.
    B, H, L, S, E = 1, 2, 4, 4, 8
    model = _model(
        [
            helper.make_node("MatMul", ["q", "kt"], ["s1"]),
            helper.make_node("Softmax", ["s1"], ["p1"], axis=-1),
            helper.make_node("MatMul", ["p1", "v1"], ["out1"]),
            helper.make_node("Softmax", ["out1"], ["p2"], axis=-1),
            helper.make_node("MatMul", ["p2", "v2"], ["out2"]),
        ],
        [
            _vi("q", [B, H, L, S]),
            _vi("kt", [B, H, S, S]),
            _vi("v1", [B, H, S, S]),
            _vi("v2", [B, H, S, E]),
        ],
        [_vi("out2", [B, H, L, E])],
    )
    fused = fuse_attention(model)
    assert len(_sdpa_nodes(fused)) == 1
    _assert_no_dangling_inputs(fused)


# ---------------------------------------------------------------------------
# Emitted IR (conversion only, no runtime)
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_composite_declaration_emitted():
    program = coreai_onnx.convert(_yolo_model())
    text = str(program)
    assert 'composite_declaration<"scaled_dot_product_attention"' in text
    assert "coreai.invoke" in text
    assert "noinline" in text


@pytest.mark.ir
def test_every_operand_gets_materialization_barrier():
    # GPU/ANE mis-execute or crash on various composite operand producers;
    # each operand must pass through the pad+slice barrier (see
    # _attention.py::_materialize) — one slice per q/k/v.
    program = coreai_onnx.convert(_yolo_model())
    text = str(program)
    assert text.count("coreai.slice") >= 3
    assert "coreai.pad" in text


@pytest.mark.ir
def test_handwritten_node_rejects_mismatched_head_dims():
    node = helper.make_node(
        "ScaledDotProductAttention", ["q", "k", "v"], ["out"], domain="coreai"
    )
    model = _model(
        [node],
        [_vi("q", [1, 2, 5, 4]), _vi("k", [1, 2, 7, 4]), _vi("v", [1, 2, 7, 8])],
        [_vi("out", [1, 2, 5, 8])],
        with_coreai_opset=True,
    )
    with pytest.raises(ConversionError, match="zero-pad"):
        coreai_onnx.convert(model)


@pytest.mark.ir
def test_handwritten_node_rejects_rank2():
    node = helper.make_node(
        "ScaledDotProductAttention", ["q", "k", "v"], ["out"], domain="coreai"
    )
    model = _model(
        [node],
        [_vi("q", [5, 4]), _vi("k", [7, 4]), _vi("v", [7, 4])],
        [_vi("out", [5, 4])],
        with_coreai_opset=True,
    )
    with pytest.raises(ConversionError, match="rank"):
        coreai_onnx.convert(model)


# ---------------------------------------------------------------------------
# Numerical parity (Core AI runtime)
# ---------------------------------------------------------------------------


@requires_coreai_runtime
async def test_parity_yolo_orientation():
    model = _yolo_model()
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_parity_standard_with_mask():
    model = _standard_model(mask=True)
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_parity_rank3():
    model = _standard_model(rank=3)
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_parity_no_scale():
    model = _standard_model(scale_op=None)
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_parity_query_dim_larger_than_value_dim():
    # Conditional DETR cross-attention shape (dk > dv): value padded, output sliced.
    model = _standard_model(dk=8, dv=4)
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_parity_unfused_pattern_still_converts():
    model = _standard_model()
    (softmax,) = [n for n in model.graph.node if n.op_type == "Softmax"]
    softmax.attribute[0].i = 2  # not the last axis: stays a raw chain
    await assert_parity(model, _random_feeds(model))


@requires_coreai_runtime
async def test_handwritten_node_parity_with_default_scale():
    B, H, L, S, E = 1, 2, 5, 7, 8
    node = helper.make_node(
        "ScaledDotProductAttention", ["q", "k", "v"], ["out"], domain="coreai"
    )
    model = _model(
        [node],
        [_vi("q", [B, H, L, E]), _vi("k", [B, H, S, E]), _vi("v", [B, H, S, E])],
        [_vi("out", [B, H, L, E])],
        with_coreai_opset=True,
    )
    feeds = _random_feeds(model)
    (got,) = await run_aimodel(model, feeds)
    expected = _reference_sdpa(feeds["q"], feeds["k"], feeds["v"], E**-0.5)
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-4)


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _yolo_expected(feeds):
    s = (feeds["q"].astype(np.float64) @ feeds["kt"].astype(np.float64)) * 0.5
    return feeds["v"].astype(np.float64) @ np.ascontiguousarray(
        _softmax(s).swapaxes(-1, -2)
    )


def _prescaled_expected(feeds):
    q, k = feeds["qin"].astype(np.float64), feeds["kin"].astype(np.float64)
    s = 0.25 * (q @ np.ascontiguousarray(k.swapaxes(-1, -2)))
    return _softmax(s) @ feeds["vin"].astype(np.float64)


def _dk_gt_dv_model():
    # Conditional DETR cross-attention: key_dim 8 > value_dim 4. The raw chain
    # aborts the GPU MPSGraph compiler ("MLIR pass manager failed"); fusion
    # (value padded 4->8, output sliced 8->4) is what keeps it compiling.
    return _standard_model(dk=8, dv=4)


def _dk_gt_dv_expected(feeds):
    s = (feeds["q"].astype(np.float64) / 2.5) @ feeds["kt"].astype(np.float64)
    return _softmax(s) @ feeds["v"].astype(np.float64)


@requires_coreai_runtime
@pytest.mark.parametrize(
    ("build", "expected_fn"),
    [
        (_yolo_model, _yolo_expected),
        (_prescaled_model, _prescaled_expected),
        (_dk_gt_dv_model, _dk_gt_dv_expected),
    ],
    ids=["yolo-orientation", "prescaled-torch-mha", "dk-gt-dv-standard"],
)
async def test_fused_attention_on_every_compute_unit(tmp_path, build, expected_fn):
    """Regression: GPU/ANE fused kernels misbehave without the barrier.

    Without the pad+slice materialization barrier the YOLO-orientation model
    returns garbage on GPU/Neural Engine and the prescaled torch-MHA shape
    segfaults the GPU compiler (CPU stays correct), so this must run on every
    available compute unit, not just the default one.
    """
    from coreai.runtime import (
        AIModel,
        ComputeUnitKind,
        NDArray,
        SpecializationOptions,
    )

    model = build()
    assert len(_sdpa_nodes(fuse_attention(model))) == 1
    feeds = _random_feeds(model)
    expected = expected_fn(feeds)

    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    asset_path = tmp_path / "m.aimodel"
    converter.to_coreai().save_asset(asset_path)

    options = [SpecializationOptions.cpu_only()] + [
        SpecializationOptions.from_preferred_compute_unit_kind(kind)
        for kind in ComputeUnitKind.available_kinds()
    ]
    for opts in options:
        ai_model = await AIModel.load(asset_path, opts)
        fn = ai_model.load_function("main")
        out = await fn({k: NDArray(v) for k, v in feeds.items()})
        got = np.asarray(out["out"].numpy())
        np.testing.assert_allclose(
            got,
            expected,
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"wrong attention output for {opts.preferred_compute_unit_kind}",
        )


# ---------------------------------------------------------------------------
# Symbolic mask dims — must stay unfused (the lowering rejects dynamic dims)
# ---------------------------------------------------------------------------


@pytest.mark.ir
def test_no_fuse_symbolic_mask_dims():
    # A mask with a symbolic dim keeps the chain's inferred shapes static
    # (broadcast against static dims), so it used to fuse — and then the SDPA
    # lowering's materialization barrier raised on the dynamic mask operand.
    # The chain must stay unfused and the model must still convert.
    model = _standard_model(mask=True)
    (mask_vi,) = [i for i in model.graph.input if i.name == "mask"]
    dim = mask_vi.type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "batch"
    _assert_not_fused(model)
    coreai_onnx.convert(model)  # raised ConversionError when fused


# ---------------------------------------------------------------------------
# Attention inside If/Loop subgraphs
# ---------------------------------------------------------------------------


def _if_attention_model(B=1, H=2, L=4, E=8):
    """The full attention chain lives inside the If then-branch (HF optimum
    merged-decoder exports wrap the whole decoder in a top-level If)."""
    then_g = helper.make_graph(
        [
            helper.make_node("MatMul", ["q", "kt"], ["ts"]),
            helper.make_node("Softmax", ["ts"], ["tp"], axis=-1),
            helper.make_node("MatMul", ["tp", "v"], ["then_out"]),
        ],
        "then",
        [],
        [_vi("then_out", [B, H, L, E])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Relu", ["v"], ["else_out"])],
        "else",
        [],
        [_vi("else_out", [B, H, L, E])],
    )
    return _model(
        [
            helper.make_node(
                "If", ["cond"], ["out"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            _vi("q", [B, H, L, E]),
            _vi("kt", [B, H, E, L]),
            _vi("v", [B, H, L, E]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("out", [B, H, L, E])],
    )


def test_fuse_attention_inside_if_branch():
    # Raw chains inside If/Loop bodies crash the on-device GPU compiler just
    # like top-level ones, so the pass must recurse into subgraphs.
    fused = fuse_attention(_if_attention_model())
    (if_node,) = fused.graph.node
    (then_g,) = [a.g for a in if_node.attribute if a.name == "then_branch"]
    ops = [n.op_type for n in then_g.node]
    assert "ScaledDotProductAttention" in ops
    assert "Softmax" not in ops
    assert any(oi.domain == "coreai" for oi in fused.opset_import)


def _if_attention_shadowed_scale_model(B=1, H=2, L=4, E=8):
    """The then-branch scales scores by a LOCAL, non-scalar ``c`` (a real
    per-element [B,H,L,L] tensor) that shadows an outer scalar ``c``. The chain
    must NOT fuse: treating the runtime tensor as a scalar scale would bake the
    wrong (outer) constant and drop the per-element multiply."""
    then_g = helper.make_graph(
        [
            helper.make_node("MatMul", ["q", "kt"], ["ts"]),
            helper.make_node("Mul", ["ts", "c"], ["sc"]),
            helper.make_node("Softmax", ["sc"], ["tp"], axis=-1),
            helper.make_node("MatMul", ["tp", "v"], ["then_out"]),
        ],
        "then",
        [],
        [_vi("then_out", [B, H, L, E])],
        initializer=[
            numpy_helper.from_array(np.full((B, H, L, L), 0.5, dtype=np.float32), "c")
        ],
    )
    else_g = helper.make_graph(
        [helper.make_node("Relu", ["v"], ["else_out"])],
        "else",
        [],
        [_vi("else_out", [B, H, L, E])],
    )
    return _model(
        [
            helper.make_node(
                "If", ["cond"], ["out"], then_branch=then_g, else_branch=else_g
            )
        ],
        [
            _vi("q", [B, H, L, E]),
            _vi("kt", [B, H, E, L]),
            _vi("v", [B, H, L, E]),
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        ],
        [_vi("out", [B, H, L, E])],
        [numpy_helper.from_array(np.array(0.5, dtype=np.float32), "c")],  # outer scalar
    )


def test_no_fuse_attention_when_local_binding_shadows_outer_scale():
    # A subgraph name bound locally (here a non-scalar initializer; likewise a
    # Loop/Scan formal input) must hide an outer scalar of the same name, so the
    # stale outer constant cannot be mistaken for the attention scale.
    fused = fuse_attention(_if_attention_shadowed_scale_model())
    (if_node,) = fused.graph.node
    (then_g,) = [a.g for a in if_node.attribute if a.name == "then_branch"]
    ops = [n.op_type for n in then_g.node]
    assert "ScaledDotProductAttention" not in ops
    assert ops == ["MatMul", "Mul", "Softmax", "MatMul"]


@requires_coreai_runtime
@pytest.mark.parametrize("cond", [True, False])
async def test_parity_attention_inside_if_branch(cond):
    model = _if_attention_model()
    rng = np.random.default_rng(0)
    feeds = {
        name: rng.standard_normal(
            [d.dim_value for d in vi.type.tensor_type.shape.dim]
        ).astype(np.float32)
        for vi in model.graph.input
        if (name := vi.name) != "cond"
    }
    feeds["cond"] = np.array(cond)
    await assert_parity(model, feeds)


# ---------------------------------------------------------------------------
# No Softmax — the pass must not pay the shape-inference round trip
# ---------------------------------------------------------------------------


def test_fuse_attention_skips_inference_without_softmax(monkeypatch):
    # fuse_attention used to re-run whole-proto shape inference (a full
    # serialize/parse round trip of every weight byte) even when no Softmax
    # exists anywhere in the model.
    model = _model(
        [helper.make_node("Relu", ["q"], ["r"])],
        [_vi("q", [2, 3])],
        [_vi("r", [2, 3])],
    )

    def _boom(*args, **kwargs):
        raise AssertionError("shape inference must not run without a Softmax")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", _boom)
    fused = fuse_attention(model)
    assert _op_types(fused) == ["Relu"]
