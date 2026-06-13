# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fuse ONNX attention chains into ``coreai::ScaledDotProductAttention`` nodes.

The on-device GPU compiler (MetalPerformanceShadersGraph) aborts with
"MLIR pass manager failed" when it has to lower a raw scaled-dot-product
attention chain (batched MatMul -> scale -> Softmax -> batched MatMul), which
crashes any GPU consumer of the asset — e.g. Xcode's Performance report.
``coreai-torch`` avoids this by emitting attention as a single
``scaled_dot_product_attention`` composite op that the OS compiler replaces
with its fused kernel.  This pass gives ONNX models the same treatment:

    scores  = MatMul(Q, Kt)                 # Kt is K with the last two
    scaled  = Mul(scores, c) | Div(scores, c)   # optional, c a scalar const
    masked  = Add(scaled, mask)                 # optional, float additive mask
    probs   = Softmax(masked, axis=-1)
    out     = MatMul(probs, V)                  # standard orientation
            | MatMul(V, Transpose(probs))       # transposed orientation
                                                #   (e.g. Ultralytics C2PSA)

is rewritten into Transpose/Pad canonicalization nodes plus one

    out = coreai::ScaledDotProductAttention(query, key, value[, mask]) {scale}

node, lowered by ``coreai_onnx._lowerings._attention`` into the composite op.

Matching is deliberately conservative — every intermediate must have exactly
one consumer and not be a graph output, ranks must be 3 or 4, the softmax must
run over the last axis, and the query/key/value shapes must be fully static.
Anything that does not match is left untouched and converts exactly as before.
The fused kernel needs query/key/value to share a head dim, so a mismatch is
zero-padded up to E = max(key_dim, value_dim): when the query/key head dim is
smaller (YOLO attention uses key_dim < head_dim) query and key are padded —
padding contributes nothing to the dot products; when it is larger (Conditional
DETR's cross-attention uses key_dim 64 > value_dim 32) value is padded and the
extra output columns sliced back off — softmax depends only on q.k, so the
result is bit-for-bit comparable either way.

The pass expects a preprocessed model (opset >= 13 Softmax semantics).  It
rewrites the top-level graph and recurses into If/Loop subgraph bodies (HF
optimum merged-decoder exports wrap the whole decoder in a top-level If, and
raw chains in branch bodies crash the GPU compiler just like top-level ones).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .._passes._model import _infer_shapes_lite
from ._index import (
    _F32_MAX,
    _FLOAT_ELEM_TYPES,
    _graph_scalars,
    _graph_tensor_infos,
    _GraphIndex,
    _TensorInfo,
)
from ._rewrite import (
    _apply_matches,
    _build_consumers,
    _collect_names,
    _drop_shadowed_scalars,
    _greedy_claim,
    _local_bindings,
    _subgraphs,
)


def _swap_last_two(rank: int) -> list[int]:
    return [*range(rank - 2), rank - 1, rank - 2]


@dataclass
class _Match:
    """One matched attention chain, in canonical SDPA terms."""

    query: str  # [.., L, dk]
    key_t: str  # [.., dk, S] — transposed key, as consumed by the scores MatMul
    value: str  # [.., S, dv] (standard) or [.., dv, S] (transposed orientation)
    mask: str | None
    scale: float
    rank: int
    dk: int
    dv: int
    transposed_output: bool
    out_matmul: onnx.NodeProto  # the final MatMul; replacement nodes go here
    nodes: list[onnx.NodeProto] = field(default_factory=list)  # all matched nodes


def _match_upward(
    softmax: onnx.NodeProto, idx: _GraphIndex
) -> tuple[onnx.NodeProto, str | None, float, list[onnx.NodeProto]] | None:
    """Match the producer chain MatMul -> [Mul|Div] -> [Add] above *softmax*.

    Returns (scores_matmul, mask_name, scale, matched_nodes) or None.
    """
    matched: list[onnx.NodeProto] = []
    cur = softmax.input[0]
    if not idx.is_internal(cur):
        return None
    node = idx.by_output.get(cur)
    if node is None:
        return None

    mask: str | None = None
    if node.op_type == "Add":
        # One operand continues the chain; the other is the additive mask.
        chain = None
        for i in (0, 1):
            cand = idx.by_output.get(node.input[i])
            if (
                cand is not None
                and cand.op_type in ("MatMul", "Mul", "Div")
                and idx.is_internal(node.input[i])
            ):
                chain = node.input[i]
                mask = node.input[1 - i]
                break
        if chain is None or mask is None:
            return None
        # The mask must not broadcast the scores up to a larger shape: the
        # fused output must keep the chain operand's exact shape.
        out_dims, chain_dims = idx.dims(node.output[0]), idx.dims(chain)
        if out_dims is None or out_dims != chain_dims:
            return None
        mask_dims = idx.dims(mask)
        if mask_dims is None or len(mask_dims) > len(out_dims):
            return None
        # The lowering materializes every fused operand through its pad+slice
        # barrier, which needs fully static dims; query/key/value are checked
        # in _match, and the mask must be too or conversion of the fused node
        # hard-fails on a model that converts fine unfused.
        if not all(isinstance(d, int) for d in mask_dims):
            return None
        matched.append(node)
        cur = chain
        node = idx.by_output.get(cur)
        if node is None:  # Defensive guard for inconsistent index metadata.
            return None

    scale = 1.0
    if node.op_type in ("Mul", "Div"):
        if node.op_type == "Mul":
            s0, s1 = idx.scalar(node.input[0]), idx.scalar(node.input[1])
            if s1 is not None:
                scale, chain = s1, node.input[0]
            elif s0 is not None:
                scale, chain = s0, node.input[1]
            else:
                return None
        else:  # Div: the denominator must be a scalar constant
            s1 = idx.scalar(node.input[1])
            if s1 is None or s1 == 0.0:
                return None
            scale, chain = 1.0 / s1, node.input[0]
        if not math.isfinite(scale) or abs(scale) > _F32_MAX:
            return None
        if not idx.is_internal(chain):
            return None
        matched.append(node)
        cur = chain
        node = idx.by_output.get(cur)
        if node is None:
            return None

    if node.op_type != "MatMul":
        return None
    matched.append(node)
    return node, mask, scale, matched


def _match_downward(
    softmax: onnx.NodeProto,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[onnx.NodeProto, str, bool, list[onnx.NodeProto]] | None:
    """Match the consumer chain below *softmax*.

    Returns (out_matmul, value_name, transposed_output, matched_nodes) or None.
    """
    probs = softmax.output[0]
    if not idx.is_internal(probs):
        return None
    # is_internal guarantees one consumer overall, but it may be a subgraph
    # capture, in which case probs is absent from the top-level consumer map.
    probs_consumers = consumers.get(probs)
    if probs_consumers is None:
        return None
    consumer = probs_consumers[0]

    # Standard orientation: out = probs @ V.
    if consumer.op_type == "MatMul" and consumer.input[0] == probs:
        return consumer, consumer.input[1], False, [consumer]

    # Transposed orientation: out = V @ probs^T  (== (probs @ V^T)^T).
    if consumer.op_type == "Transpose":
        rank_dims = idx.dims(probs)
        if rank_dims is None:
            return None
        rank = len(rank_dims)
        perm = next(
            (list(a.ints) for a in consumer.attribute if a.name == "perm" and a.ints),
            list(reversed(range(rank))),  # ONNX Transpose default
        )
        if perm != _swap_last_two(rank):
            return None
        probs_t = consumer.output[0]
        if not idx.is_internal(probs_t):
            return None
        probs_t_consumers = consumers.get(probs_t)
        if probs_t_consumers is None:  # sole consumer is a subgraph capture
            return None
        matmul = probs_t_consumers[0]
        if matmul.op_type != "MatMul" or matmul.input[1] != probs_t:
            return None
        return matmul, matmul.input[0], True, [consumer, matmul]

    return None


def _match(
    softmax: onnx.NodeProto,
    idx: _GraphIndex,
    consumers: dict[str, list[onnx.NodeProto]],
) -> _Match | None:
    softmax_in = idx.infos.get(softmax.input[0])
    if softmax_in is None or softmax_in.elem_type not in _FLOAT_ELEM_TYPES:
        return None
    rank = len(softmax_in.dims)
    if rank not in (3, 4):
        return None
    axis = next((a.i for a in softmax.attribute if a.name == "axis"), -1)
    if axis % rank != rank - 1:
        return None

    up = _match_upward(softmax, idx)
    if up is None:
        return None
    scores_matmul, mask, scale, up_nodes = up

    down = _match_downward(softmax, idx, consumers)
    if down is None:
        return None
    out_matmul, value, transposed_output, down_nodes = down

    query, key_t = scores_matmul.input[0], scores_matmul.input[1]
    q_dims, kt_dims, v_dims = idx.dims(query), idx.dims(key_t), idx.dims(value)
    if q_dims is None or kt_dims is None or v_dims is None:
        return None
    if not (len(q_dims) == len(kt_dims) == len(v_dims) == rank):
        return None
    # Fully static shapes only: the canonicalization (zero-padding and the
    # materialization barrier in the lowering) needs them, and Core AI assets
    # are specialized for fixed shapes anyway.  Dynamic-shape attention is
    # left unfused.
    if not all(isinstance(d, int) for dims in (q_dims, kt_dims, v_dims) for d in dims):
        return None
    # Conservative: identical leading (batch/head) dims, no broadcasting.
    if not (q_dims[:-2] == kt_dims[:-2] == v_dims[:-2]):
        return None
    dk = q_dims[-1]
    dv = v_dims[-2] if transposed_output else v_dims[-1]
    if not isinstance(dk, int) or not isinstance(dv, int):  # mypy narrowing
        return None
    # dk != dv is fine: the kernel needs a shared head dim E = max(dk, dv), and
    # _replacement_nodes zero-pads the smaller side up to it. Padding query/key
    # (dk < dv) leaves the dot products unchanged; padding value (dk > dv) only
    # appends zero output columns, which the lowering slices back off.

    return _Match(
        query=query,
        key_t=key_t,
        value=value,
        mask=mask,
        scale=scale,
        rank=rank,
        dk=dk,
        dv=dv,
        transposed_output=transposed_output,
        out_matmul=out_matmul,
        nodes=[*up_nodes, softmax, *down_nodes],
    )


def _replacement_nodes(
    match: _Match, graph: onnx.GraphProto, used_names: set[str]
) -> list[onnx.NodeProto]:
    """Build Transpose/Pad canonicalization + the fused node for *match*."""

    def unique(base: str) -> str:
        name = base
        i = 1
        while name in used_names:
            name = f"{base}_{i}"
            i += 1
        used_names.add(name)
        return name

    out_name = match.out_matmul.output[0]
    prefix = f"{out_name}/sdpa"
    perm = _swap_last_two(match.rank)
    nodes: list[onnx.NodeProto] = []

    def pad_last_dim(src: str, amount: int) -> str:
        """Append *amount* zero columns to *src*'s last dim; return the new name."""
        pads = np.zeros(2 * match.rank, dtype=np.int64)
        pads[-1] = amount
        pads_name = unique(f"{prefix}_pads")
        graph.initializer.append(numpy_helper.from_array(pads, name=pads_name))
        padded = unique(f"{prefix}_pad")
        nodes.append(helper.make_node("Pad", [src, pads_name], [padded], name=padded))
        return padded

    query = match.query
    key = unique(f"{prefix}_key")
    nodes.append(
        helper.make_node("Transpose", [match.key_t], [key], perm=perm, name=key)
    )

    # The fused kernel needs query/key/value to share head dim E = max(dk, dv);
    # zero-pad the smaller side up to it.
    if match.dk < match.dv:
        # Zero-pad query/key up to dv: zeros do not contribute to the q.k dot
        # products, so the scores (and thus the output) are unchanged.
        query = pad_last_dim(query, match.dv - match.dk)
        key = pad_last_dim(key, match.dv - match.dk)

    value = match.value
    if match.transposed_output:
        value = unique(f"{prefix}_value")
        nodes.append(
            helper.make_node("Transpose", [match.value], [value], perm=perm, name=value)
        )
    if match.dk > match.dv:
        # Zero-pad value up to dk. softmax depends only on q.k, so the extra
        # value columns are zero and produce zero output columns, sliced off
        # below — leaving the dv-wide result bit-for-bit unchanged.
        value = pad_last_dim(value, match.dk - match.dv)

    # The fused node yields E = max(dk, dv) columns; restore dv when value grew.
    final_out = unique(f"{prefix}_out") if match.transposed_output else out_name
    fused_out = unique(f"{prefix}_wide") if match.dk > match.dv else final_out
    inputs = [query, key, value] + ([match.mask] if match.mask is not None else [])
    nodes.append(
        helper.make_node(
            "ScaledDotProductAttention",
            inputs,
            [fused_out],
            domain="coreai",
            scale=float(match.scale),
            name=unique(f"{prefix}_fused"),
        )
    )
    if match.dk > match.dv:
        starts = unique(f"{prefix}_starts")
        ends = unique(f"{prefix}_ends")
        axes = unique(f"{prefix}_axes")
        graph.initializer.append(
            numpy_helper.from_array(np.array([0], np.int64), starts)
        )
        graph.initializer.append(
            numpy_helper.from_array(np.array([match.dv], np.int64), ends)
        )
        graph.initializer.append(
            numpy_helper.from_array(np.array([match.rank - 1], np.int64), axes)
        )
        nodes.append(
            helper.make_node(
                "Slice", [fused_out, starts, ends, axes], [final_out], name=final_out
            )
        )
    if match.transposed_output:
        nodes.append(
            helper.make_node(
                "Transpose", [final_out], [out_name], perm=perm, name=out_name
            )
        )
    return nodes


def _has_softmax(graph: onnx.GraphProto) -> bool:
    return any(
        n.op_type == "Softmax" or any(_has_softmax(sg) for sg in _subgraphs(n))
        for n in graph.node
    )


def _fuse_graph(
    graph: onnx.GraphProto,
    outer_infos: dict[str, _TensorInfo],
    outer_scalars: dict[str, np.ndarray],
    used_names: set[str],
) -> bool:
    """Fuse attention chains in *graph*, then recurse into If/Loop bodies.

    Outer-scope infos/scalars stay visible to subgraphs (chains may consume
    captured values); local entries shadow them.  Chains never span graph
    boundaries: a chain node produced in an outer scope has no local producer
    entry, so the matcher bails out.
    """
    local_infos = _graph_tensor_infos(graph)
    local_scalars = _graph_scalars(graph)
    infos = {**outer_infos, **local_infos}
    scalars = {**outer_scalars, **local_scalars}
    # A name bound locally in this scope shadows an outer entry captured by the
    # same name; the local binding is a runtime value here, so the stale outer
    # constant/shape must not be visible to matchers (it would bake a wrong scale
    # or pass a bogus shape check). Local bindings are node outputs, subgraph
    # formal inputs (Loop/Scan body params), and initializers. local_infos
    # already re-supplies graph inputs/initializers (so they override outer infos
    # via the merge above); only node outputs absent from it need the stale outer
    # info dropped. For scalars only size-1 initializers survive (kept in
    # local_scalars), so every other local binding drops its outer scalar.
    bindings = _local_bindings(graph)
    for n in graph.node:
        for out in n.output:
            if out not in local_infos:
                infos.pop(out, None)
    _drop_shadowed_scalars(scalars, bindings, local_scalars)
    idx = _GraphIndex.build(graph, infos, scalars)
    consumers = _build_consumers(graph)

    matches = _greedy_claim(
        graph, lambda n: _match(n, idx, consumers) if n.op_type == "Softmax" else None
    )
    if matches:
        _apply_matches(
            graph,
            matches,
            anchor=lambda m: m.out_matmul,
            expand=lambda m: _replacement_nodes(m, graph, used_names),
        )

    fused = bool(matches)
    for n in graph.node:
        for sg in _subgraphs(n):
            fused = _fuse_graph(sg, infos, scalars, used_names) or fused
    return fused


def fuse_attention(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite attention chains in *model* (and its If/Loop subgraphs) into
    fused SDPA nodes."""
    # No Softmax anywhere -> nothing can match; skip the shape-inference
    # round trip (it copies the full model, value_info from preprocess stays).
    if not _has_softmax(model.graph):
        return model
    model = _infer_shapes_lite(model)
    used_names: set[str] = set()
    _collect_names(model.graph, used_names)
    if _fuse_graph(model.graph, {}, {}, used_names) and not any(
        oi.domain == "coreai" for oi in model.opset_import
    ):
        model.opset_import.append(helper.make_opsetid("coreai", 1))
    return model
