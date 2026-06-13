# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for ONNX control-flow ops (If).

Control-flow lowerings differ from the rest of the registry: they declare a
4th positional parameter, ``converter``, and ``OnnxConverter._lower_node``
detects this (via ``inspect.signature``, cached) and passes itself in. The
converter is needed to recursively lower the branch GraphProtos with
``converter._lower_graph_body`` — which also makes nested If work for free.
"""

from collections.abc import Callable, Sequence
from typing import Any, cast

import onnx

from .._ir import Location, Type, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._type_mapping import tensor_type_from_value_info
from .._utils import operands

_IfRegions = tuple[Value | Sequence[Value], Sequence[Any]]


def _coreai_if_with_regions(results: Sequence[Type], condition: Value) -> _IfRegions:
    """Call coreai.if_ through the runtime contract missing from its stub."""
    return cast("_IfRegions", coreai.if_(results=results, condition=condition))


def replace_if(
    values_map: dict[str, Value],
    node: onnx.NodeProto,
    loc: Location,
    converter: Any,
) -> list[Value]:
    (cond,) = operands(values_map, node, [0])
    # coreai.if_ wants a scalar i1 condition; ONNX allows a 1-element tensor.
    if hasattr(cond.type, "rank") and (tensor_type(cond).rank or 0) > 0:
        cond = coreai.shrink_dims(cond, list(range(tensor_type(cond).rank)))

    # Branch subgraphs come through as raw GraphProtos on node.attribute.
    branches: dict[str, onnx.GraphProto] = {
        a.name: a.g for a in node.attribute if a.type == onnx.AttributeProto.GRAPH
    }
    try:
        then_g, else_g = branches["then_branch"], branches["else_branch"]
    except KeyError as exc:
        raise ValueError(f"If: missing {exc.args[0]} attribute") from None

    if len(then_g.output) != len(else_g.output):
        raise ValueError(
            f"If: then_branch yields {len(then_g.output)} output(s) but "
            f"else_branch yields {len(else_g.output)}"
        )

    # Result types from the branch output value_infos (shape inference in
    # preprocess populates them); fall back to the other branch per output.
    # coreai.if requires both regions to yield exactly the declared result
    # type, so ONNX-legal branches with differing shapes cannot be lowered —
    # reject them here instead of emitting IR that fails at save_asset.
    result_types = []
    for then_vi, else_vi in zip(then_g.output, else_g.output, strict=True):
        then_tt, else_tt = then_vi.type.tensor_type, else_vi.type.tensor_type
        vi = then_vi if then_tt.elem_type else else_vi
        if not vi.type.tensor_type.elem_type:
            raise ValueError(
                f"If: branch output '{then_vi.name}' has no type information; "
                "cannot determine the result type"
            )
        if then_tt.elem_type and else_tt.elem_type:
            if then_tt.elem_type != else_tt.elem_type:
                raise ValueError(
                    f"If: branch output '{then_vi.name}' element type differs "
                    "between then/else branches"
                )
            then_dims = [
                d.dim_value if d.HasField("dim_value") else None
                for d in then_tt.shape.dim
            ]
            else_dims = [
                d.dim_value if d.HasField("dim_value") else None
                for d in else_tt.shape.dim
            ]
            if then_dims != else_dims:
                raise ValueError(
                    f"If: branch output '{then_vi.name}' shapes differ between "
                    f"branches ({then_dims} vs {else_dims}); coreai.if requires "
                    "matching static branch output shapes"
                )
        if not vi.type.tensor_type.HasField("shape"):
            # No shape field means unknown rank, not rank 0.
            raise ValueError(
                f"If: branch output '{then_vi.name}' has unknown rank; "
                "cannot determine the result type"
            )
        result_types.append(tensor_type_from_value_info(vi))

    # The upstream wrapper types every op's return as Value, but coreai.if_
    # actually returns a (results, region_builders) pair.
    if_results, region_builders = _coreai_if_with_regions(result_types, cond)

    for builder, branch in zip(region_builders, (then_g, else_g), strict=True):
        # Child map: outer names stay visible (captured by name per ONNX
        # scoping rules); branch-local names do not leak back out.
        branch_values = dict(values_map)
        with builder:
            converter._lower_graph_body(branch, branch_values, loc)
            yielded = []
            for out in branch.output:
                if out.name not in branch_values:
                    raise ValueError(
                        f"If: branch '{branch.name}' output '{out.name}' was "
                        "not produced by any node"
                    )
                yielded.append(branch_values[out.name])
            coreai.yield_(yielded)

    return [if_results] if isinstance(if_results, Value) else list(if_results)


REGISTRY: dict[str, Callable[..., Any]] = {
    "If": replace_if,
}
