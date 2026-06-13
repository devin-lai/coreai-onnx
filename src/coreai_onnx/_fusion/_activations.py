# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Activation fusion: decomposed GELU and SiLU.

The analogous (much simpler) pattern pass to ``fuse_attention`` for
decomposed GELU/SiLU chains.
"""

import math
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from ._index import _graph_scalars, _GraphIndex
from ._rewrite import (
    _apply_matches,
    _build_consumers,
    _drop_shadowed_scalars,
    _greedy_claim,
    _local_bindings,
    _subgraphs,
)

_SQRT2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_TANH_CUBE = 0.044715  # the constant baked into the tanh approximation


def _scalar_close(idx: _GraphIndex, name: str, target: float) -> bool:
    val = idx.scalar(name)
    # rel_tol covers f32-rounded exporter constants (e.g. 1.4142135381698608).
    return val is not None and math.isclose(val, target, rel_tol=1e-5)


def _other_input(node: onnx.NodeProto, name: str) -> str | None:
    """The other operand of a two-input node, or None."""
    if len(node.input) != 2:
        return None
    if node.input[0] == name:
        return node.input[1]
    if node.input[1] == name:
        return node.input[0]
    return None


def _sole_consumer(
    idx: _GraphIndex, consumers: dict[str, list[onnx.NodeProto]], name: str
) -> onnx.NodeProto | None:
    """The unique top-level consumer of internal value *name*, or None."""
    if not idx.is_internal(name):
        return None
    found = consumers.get(name)
    # found is None when the sole consumer is a subgraph capture.
    return found[0] if found else None


@dataclass
class _ActMatch:
    """One matched activation chain."""

    final: onnx.NodeProto  # node whose output the fused node replaces
    x: str
    op_type: str
    domain: str
    attrs: dict
    nodes: list[onnx.NodeProto]


def _match_gelu_tail(
    act: onnx.NodeProto,
    x: str,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[onnx.NodeProto, list[onnx.NodeProto]] | None:
    """Match Add(act_out, 1) followed by the two-Mul product with {x, 0.5}.

    Handles every operand arrangement: Mul(Mul(x, add), 0.5),
    Mul(Mul(add, 0.5), x), and Mul(add, Mul(x, 0.5)).
    Returns (final_node, matched_nodes) or None.
    """
    add = _sole_consumer(idx, consumers, act.output[0])
    if add is None or add.op_type != "Add":
        return None
    one = _other_input(add, act.output[0])
    if one is None or not _scalar_close(idx, one, 1.0):
        return None
    m1 = _sole_consumer(idx, consumers, add.output[0])
    if m1 is None or m1.op_type != "Mul":
        return None
    t = _other_input(m1, add.output[0])
    if t is None:
        return None

    def half_x(a: str, b: str) -> bool:
        return (a == x and _scalar_close(idx, b, 0.5)) or (
            b == x and _scalar_close(idx, a, 0.5)
        )

    m2 = _sole_consumer(idx, consumers, m1.output[0])
    if m2 is not None and m2.op_type == "Mul":
        u = _other_input(m2, m1.output[0])
        if u is not None and half_x(t, u):
            return m2, [add, m1, m2]
    inner = idx.by_output.get(t)
    if (
        inner is not None
        and inner.op_type == "Mul"
        and idx.is_internal(t)
        and len(inner.input) == 2
        and half_x(inner.input[0], inner.input[1])
    ):
        return m1, [add, inner, m1]
    return None


def _match_gelu_erf(
    erf: onnx.NodeProto,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> _ActMatch | None:
    """x*0.5*(1 + Erf(x/sqrt(2))) — torch's opset-17 nn.GELU export shape."""
    if not idx.is_internal(erf.input[0]):
        return None
    pre = idx.by_output.get(erf.input[0])
    if pre is None:
        return None
    x = None
    if pre.op_type == "Div" and _scalar_close(idx, pre.input[1], _SQRT2):
        x = pre.input[0]
    elif pre.op_type == "Mul":
        for i in (0, 1):
            if _scalar_close(idx, pre.input[i], 1.0 / _SQRT2):
                x = pre.input[1 - i]
                break
    if x is None:
        return None
    tail = _match_gelu_tail(erf, x, idx, consumers)
    if tail is None:
        return None
    final, tail_nodes = tail
    return _ActMatch(
        final, x, "Gelu", "", {"approximate": "none"}, [pre, erf, *tail_nodes]
    )


def _match_cube(name: str, x: str, idx: _GraphIndex) -> list[onnx.NodeProto] | None:
    """Nodes computing *name* == x**3: Pow(x, 3) or x * (x * x)."""
    if not idx.is_internal(name):
        return None
    n = idx.by_output.get(name)
    if n is None:
        return None
    if n.op_type == "Pow":
        if n.input[0] == x and _scalar_close(idx, n.input[1], 3.0):
            return [n]
        return None
    if n.op_type != "Mul":
        return None
    sq = _other_input(n, x)
    if sq is None or sq == x or not idx.is_internal(sq):
        return None
    sq_node = idx.by_output.get(sq)
    if sq_node is None or sq_node.op_type != "Mul":
        return None
    if sq_node.input[0] == x and sq_node.input[1] == x:
        return [sq_node, n]
    return None


def _match_gelu_tanh(
    tanh: onnx.NodeProto,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> _ActMatch | None:
    """0.5*x*(1+Tanh(sqrt(2/pi)*(x + 0.044715*x^3))) — tanh-approximate GELU."""
    if not idx.is_internal(tanh.input[0]):
        return None
    pre = idx.by_output.get(tanh.input[0])
    if pre is None or pre.op_type != "Mul":
        return None
    inner_name = None
    for i in (0, 1):
        if _scalar_close(idx, pre.input[i], _SQRT_2_OVER_PI):
            inner_name = pre.input[1 - i]
            break
    if inner_name is None or not idx.is_internal(inner_name):
        return None
    add = idx.by_output.get(inner_name)
    if add is None or add.op_type != "Add":
        return None
    for i in (0, 1):
        x, corr = add.input[i], add.input[1 - i]
        if not idx.is_internal(corr):
            continue
        cube_mul = idx.by_output.get(corr)
        if cube_mul is None or cube_mul.op_type != "Mul":
            continue
        cube = None
        for j in (0, 1):
            if _scalar_close(idx, cube_mul.input[j], _GELU_TANH_CUBE):
                cube = cube_mul.input[1 - j]
                break
        if cube is None:
            continue
        cube_nodes = _match_cube(cube, x, idx)
        if cube_nodes is None:
            continue
        tail = _match_gelu_tail(tanh, x, idx, consumers)
        if tail is None:
            return None
        final, tail_nodes = tail
        return _ActMatch(
            final,
            x,
            "Gelu",
            "",
            {"approximate": "tanh"},
            [*cube_nodes, cube_mul, add, pre, tanh, *tail_nodes],
        )
    return None


def _match_silu(
    sigmoid: onnx.NodeProto,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> _ActMatch | None:
    """Sigmoid(x) * x — how every exporter encodes SiLU/Swish (no ONNX op)."""
    mul = _sole_consumer(idx, consumers, sigmoid.output[0])
    if mul is None or mul.op_type != "Mul":
        return None
    if _other_input(mul, sigmoid.output[0]) != sigmoid.input[0]:
        return None
    return _ActMatch(mul, sigmoid.input[0], "Silu", "coreai_onnx", {}, [sigmoid, mul])


_ACT_MATCHERS = {
    "Erf": _match_gelu_erf,
    "Tanh": _match_gelu_tanh,
    "Sigmoid": _match_silu,
}


def _fuse_activation_graph(
    graph: onnx.GraphProto, outer_scalars: dict[str, np.ndarray]
) -> bool:
    """Rewrite decomposed activation chains in *graph* and nested subgraphs.

    Returns True when a custom-domain ``coreai_onnx`` node was emitted and the
    model needs the corresponding opset import.
    """
    local_scalars = _graph_scalars(graph)
    scalars = {**outer_scalars, **local_scalars}
    # A name bound locally in this scope shadows an outer initializer captured by
    # the same name. When it does, the local binding is a runtime value here, so
    # the stale outer scalar must not be visible to matchers (it would bake a
    # wrong constant into a fused activation). Local bindings are node outputs,
    # subgraph formal inputs (Loop/Scan body params), and initializers; only
    # size-1 initializers survive as scalars (they are kept in local_scalars).
    _drop_shadowed_scalars(scalars, _local_bindings(graph), local_scalars)
    idx = _GraphIndex.build(graph, {}, scalars)
    consumers = _build_consumers(graph)

    def match_one(n):
        matcher = _ACT_MATCHERS.get(n.op_type)
        return None if matcher is None else matcher(n, idx, consumers)

    matches = _greedy_claim(graph, match_one)
    needs_custom_opset = any(m.domain == "coreai_onnx" for m in matches)

    def expand(m):
        out = m.final.output[0]
        return [
            helper.make_node(
                m.op_type,
                [m.x],
                [out],
                domain=m.domain,
                name=f"{out}/{m.op_type.lower()}_fused",
                **m.attrs,
            )
        ]

    if matches:
        _apply_matches(graph, matches, anchor=lambda m: m.final, expand=expand)

    for n in graph.node:
        for sg in _subgraphs(n):
            needs_custom_opset = (
                _fuse_activation_graph(sg, scalars) or needs_custom_opset
            )
    return needs_custom_opset


def fuse_activations(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite decomposed GELU/SiLU chains in *model*'s graphs.

    ONNX has no SiLU op (through opset 22) and Gelu only exists from opset 20,
    so torch exports at the opset-17 baseline decompose both; the coreai
    dialect ships fused gelu/silu primitives the on-device compilers never
    reconstruct from the decomposed form.  Matched GELU chains collapse to a
    single ONNX Gelu node (with the matching 'approximate' attribute, lowered
    by replace_gelu); SiLU becomes a custom ``coreai_onnx::Silu`` node.  The
    Gelu node deliberately ignores the model's declared opset: the model was
    already validated and the lowering registry keys on op_type alone.

    Matching mirrors fuse_attention's conservatism: every intermediate must
    have exactly one consumer and not be a graph output, and scalar constants
    must match the decomposition's within tolerance. The pass recurses into
    If/Loop subgraphs so branch-wrapped exports get the same fused CoreAI
    primitives as top-level graphs.
    """
    if _fuse_activation_graph(model.graph, {}) and not any(
        oi.domain == "coreai_onnx" for oi in model.opset_import
    ):
        model.opset_import.append(helper.make_opsetid("coreai_onnx", 1))
    return model
