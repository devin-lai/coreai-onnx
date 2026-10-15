# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Cleanup passes: Identity removal, dead-node and dead-initializer pruning."""

import onnx

from .._utils import (
    apply_graphs_bottomup,
    apply_graphs_topdown,
    iter_node_subgraphs,
    iter_subgraph_inputs,
)


def _bound_names(graph: onnx.GraphProto) -> set[str]:
    return (
        {vi.name for vi in graph.input}
        | {init.name for init in graph.initializer}
        | {out for node in graph.node for out in node.output}
    )


def _subgraph_scopes(graph: onnx.GraphProto) -> list[tuple[set[str], set[str]]]:
    """Local bindings and possible outer captures for each descendant scope."""
    scopes = []
    for node in graph.node:
        for sub in iter_node_subgraphs(node):
            bound = _bound_names(sub)
            used = {o.name for o in sub.output}
            for sn in sub.node:
                used.update(sn.input)
                used.update(iter_subgraph_inputs(sn))
            scopes.append((bound, used - bound))
            scopes.extend(_subgraph_scopes(sub))
    return scopes


def _rewrite_subgraph_inputs(node: onnx.NodeProto, rename: dict[str, str]) -> None:
    """Recursively rewrite subgraph node inputs using *rename*, respecting scope.

    A subgraph that rebinds a name — as a formal input (Loop/Scan body
    parameter), a local initializer, or a node output — shadows the outer value
    of that name; references to it inside the subgraph resolve to the local
    binding and must NOT be rewritten. Each subgraph therefore rewrites with the
    map narrowed to drop the names it rebinds, and recurses with that narrowed
    map (a deeper subgraph may rebind yet more names). Without this, removing an
    outer ``Identity(x) -> t`` would rewrite a subgraph's own ``t`` to ``x`` and
    silently miscompile a checker-valid model.
    """
    for graph in iter_node_subgraphs(node):
        bound = _bound_names(graph)
        local = {k: v for k, v in rename.items() if k not in bound}
        if not local:
            continue
        for sn in graph.node:
            for i, name in enumerate(sn.input):
                if name in local:
                    sn.input[i] = local[name]
            _rewrite_subgraph_inputs(sn, local)


def _remove_identity_graph(g: onnx.GraphProto) -> None:
    output_names = {o.name for o in g.output}
    scopes = _subgraph_scopes(g)
    rename: dict[str, str] = {}
    kept = []
    for node in g.node:
        if (
            node.domain in ("", "ai.onnx")
            and node.op_type == "Identity"
            and node.output[0] not in output_names
        ):
            source = rename.get(node.input[0], node.input[0])
            # Renaming an outer capture to a locally bound name would change
            # which tensor the branch reads. Retain the alias in that case.
            if not any(
                source in bound and node.output[0] in captures
                for bound, captures in scopes
            ):
                rename[node.output[0]] = source
                continue
        kept.append(node)
    for node in kept:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
        # C1: also rewrite inputs inside subgraph nodes, recursively
        _rewrite_subgraph_inputs(node, rename)
    del g.node[:]
    g.node.extend(kept)


def remove_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    apply_graphs_topdown(model.graph, _remove_identity_graph)
    return model


def _eliminate_dead_nodes_graph(g: onnx.GraphProto) -> None:
    needed = {o.name for o in g.output}
    kept_rev = []
    for node in reversed(g.node):
        if any(o in needed for o in node.output):
            kept_rev.append(node)
            needed.update(node.input)
            # C3: subgraphs capture outer values by name — scan recursively
            needed.update(iter_subgraph_inputs(node))
    del g.node[:]
    g.node.extend(reversed(kept_rev))


def eliminate_dead_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    apply_graphs_bottomup(model.graph, _eliminate_dead_nodes_graph)
    return model


def _prune_dead_initializers_graph(g: onnx.GraphProto) -> None:
    """Drop initializers no node input, subgraph capture, or graph output uses.

    fold_constants inlines folded results as new initializers but leaves the
    consumed sources behind (e.g. the f32 copy of a Cast-to-f16 weight), and
    the converter emits a coreai.constant per initializer.  Initializers that
    shadow graph inputs are kept: their names define the converted signature
    (the shadowed input is dropped — ONNX default-value semantics).
    """
    used = (
        {i for n in g.node for i in n.input}
        | {name for n in g.node for name in iter_subgraph_inputs(n)}
        | {o.name for o in g.output}
        | {vi.name for vi in g.input}
    )
    for i in reversed(range(len(g.initializer))):
        if g.initializer[i].name not in used:
            del g.initializer[i]


def prune_dead_initializers(model: onnx.ModelProto) -> onnx.ModelProto:
    apply_graphs_bottomup(model.graph, _prune_dead_initializers_graph)
    return model
