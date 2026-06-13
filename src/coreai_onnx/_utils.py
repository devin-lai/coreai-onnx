# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared helpers for lowering functions."""

from collections.abc import Callable, Iterator
from typing import Any

import onnx
from onnx import numpy_helper

from ._ir import Value
from ._type_mapping import narrow_array


def op_key(node: onnx.NodeProto) -> str:
    """Registry key: bare op_type for the default domain, 'domain::OpType' otherwise."""
    if node.domain in ("", "ai.onnx"):
        return node.op_type
    return f"{node.domain}::{node.op_type}"


def iter_graph_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    """Yield every node in *graph*, recursing into GRAPH/GRAPHS subgraph
    attributes (If/Loop/Scan bodies). The single place that knows how nodes
    nest, so coverage analysis, the conversion coverage gate, and preprocessing
    all traverse identically and stay in step with future control-flow ops."""
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from iter_graph_nodes(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    yield from iter_graph_nodes(sub)


def attrs(node: onnx.NodeProto) -> dict[str, Any]:
    """Node attributes as plain Python values (strings decoded, tensors → numpy)."""
    out: dict[str, Any] = {}
    for a in node.attribute:
        v = onnx.helper.get_attribute_value(a)
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        elif isinstance(v, onnx.TensorProto):
            v = narrow_array(numpy_helper.to_array(v), context=f"attr '{a.name}'")
        elif isinstance(v, list) and v and isinstance(v[0], bytes):
            v = [b.decode("utf-8") for b in v]
        out[a.name] = v
    return out


def operand(values_map: dict[str, Value], node: onnx.NodeProto, i: int) -> Value | None:
    """The i-th input Value, or None if absent/optional (empty name)."""
    if i >= len(node.input) or node.input[i] == "":
        return None
    name = node.input[i]
    if name not in values_map:
        raise ValueError(
            f"node '{node.name}': input '{name}' does not resolve to a known value"
        )
    return values_map[name]


def operands(
    values_map: dict[str, Value], node: onnx.NodeProto, idx: list[int]
) -> list[Value]:
    out = []
    for i in idx:
        v = operand(values_map, node, i)
        if v is None:
            raise ValueError(f"node '{node.name}': required input {i} is missing")
        out.append(v)
    return out


def normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def iter_node_subgraphs(node: onnx.NodeProto) -> Iterator[onnx.GraphProto]:
    """Yield GRAPH/GRAPHS attributes directly attached to *node*."""
    for a in node.attribute:
        if a.type == onnx.AttributeProto.GRAPH:
            yield a.g
        elif a.type == onnx.AttributeProto.GRAPHS:
            yield from a.graphs


def iter_subgraph_inputs(node: onnx.NodeProto) -> Iterator[str]:
    """Recursively yield all inputs of nodes inside GRAPH attributes of *node*."""
    for graph in iter_node_subgraphs(node):
        for sn in graph.node:
            yield from sn.input
            yield from iter_subgraph_inputs(sn)


def apply_graphs_topdown(
    graph: onnx.GraphProto, fn: Callable[[onnx.GraphProto], None]
) -> None:
    """Apply *fn* to *graph*, then to every nested subgraph (parents first)."""
    fn(graph)
    for node in graph.node:
        for subgraph in iter_node_subgraphs(node):
            apply_graphs_topdown(subgraph, fn)


def apply_graphs_bottomup(
    graph: onnx.GraphProto, fn: Callable[[onnx.GraphProto], None]
) -> None:
    """Apply *fn* to every nested subgraph, then to *graph* (children first)."""
    for node in graph.node:
        for subgraph in iter_node_subgraphs(node):
            apply_graphs_bottomup(subgraph, fn)
    fn(graph)
