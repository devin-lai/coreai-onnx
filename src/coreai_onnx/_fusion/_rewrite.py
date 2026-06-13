# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared graph-rewrite plumbing for the fusion passes."""

import numpy as np
import onnx

from .._utils import iter_node_subgraphs as _subgraphs


def _collect_names(graph: onnx.GraphProto, out: set[str]) -> None:
    out.update(i.name for i in graph.initializer)
    out.update(vi.name for vi in (*graph.input, *graph.output, *graph.value_info))
    for n in graph.node:
        out.update(n.input)
        out.update(n.output)
        for sg in _subgraphs(n):
            _collect_names(sg, out)


def _purge_stale_value_info(graph: onnx.GraphProto, removed: set[str]) -> None:
    """Drop value_info entries for intermediates a fusion rewrite deleted.

    Chain outputs that survive (the match's final output keeps its name) stay
    annotated; only entries naming values no node produces anymore go away."""
    stale = removed - {out for n in graph.node for out in n.output}
    if not stale:
        return
    kept = [vi for vi in graph.value_info if vi.name not in stale]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)


def _build_consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """name -> nodes consuming it, in graph order."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for n in graph.node:
        for name in n.input:
            consumers.setdefault(name, []).append(n)
    return consumers


def _local_bindings(graph: onnx.GraphProto) -> set[str]:
    """Names bound in this scope: node outputs, formal inputs (Loop/Scan body
    params), and initializers. A local binding shadows an outer capture of the
    same name, so the stale outer constant must not be visible to matchers."""
    return (
        {out for n in graph.node for out in n.output}
        | {vi.name for vi in graph.input}
        | {init.name for init in graph.initializer}
    )


def _drop_shadowed_scalars(
    scalars: dict[str, np.ndarray],
    bindings: set[str],
    local_scalars: dict[str, np.ndarray],
) -> None:
    """For scalars only size-1 initializers survive (kept in local_scalars);
    every other local binding drops its outer scalar. Mutates *scalars*."""
    for name in bindings:
        if name not in local_scalars:
            scalars.pop(name, None)


def _greedy_claim(graph: onnx.GraphProto, match_one) -> list:
    """Greedy in graph order: a chain whose nodes were already claimed by an
    earlier match (e.g. stacked attention sharing its boundary MatMul) is
    left unfused — fusing both would delete nodes the later rewrite still
    references. match_one(node) returns a match (with .nodes) or None."""
    matches = []
    claimed: set[int] = set()
    for n in graph.node:
        m = match_one(n)
        if m is None or any(id(node) in claimed for node in m.nodes):
            continue
        matches.append(m)
        claimed.update(id(node) for node in m.nodes)
    return matches


def _apply_matches(graph: onnx.GraphProto, matches, anchor, expand) -> None:
    """Replace each match's node set in-place: at anchor(m)'s position splice
    in expand(m); drop the other matched nodes; purge stale value_info."""
    matched_ids = {id(n) for m in matches for n in m.nodes}
    expansions = {id(anchor(m)): m for m in matches}
    new_nodes: list[onnx.NodeProto] = []
    for n in graph.node:
        m = expansions.get(id(n))
        if m is not None:
            new_nodes.extend(expand(m))
        elif id(n) not in matched_ids:
            new_nodes.append(n)
    del graph.node[:]
    graph.node.extend(new_nodes)
    _purge_stale_value_info(
        graph, {out for m in matches for n in m.nodes for out in n.output}
    )
