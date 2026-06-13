# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constant folding: evaluate statically-known foldable nodes and inline them."""

import warnings

import numpy as np
import onnx
from onnx import numpy_helper

from .._utils import apply_graphs_topdown, iter_subgraph_inputs

# Ops we attempt to fold when all inputs are statically known.
_FOLDABLE = {
    "Shape",
    "Size",
    "Gather",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Slice",
    "Cast",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Range",
    "ConstantOfShape",
    # torch's MultiheadAttention export computes Slice starts/ends via
    # Mod/Reshape chains over constants; fold them so Slice sees constants.
    "Mod",
    "Reshape",
    "Constant",  # C4: non-tensor variants fall through to ReferenceEvaluator
    # Constant mask/shape pipelines from torch/HF exports: Equal/Where build
    # additive attention masks, Expand/Tile/Transpose appear over constants.
    "Equal",
    "Where",
    "Expand",
    "Tile",
    "Transpose",
    "Neg",
    "Floor",
    "Ceil",
    "ReduceProd",
}

# Results (or inputs) above this size are not folded: ConstantOfShape/Range/
# Expand can blow a few bytes of shape data up into dense masks, and folding
# those bloats the proto and bakes the mask into the asset (mirrors
# onnx-simplifier's large-tensor guard).
_FOLD_SIZE_LIMIT = 1 << 24  # 16 MiB


def _fold_constants_graph(
    model: onnx.ModelProto, g: onnx.GraphProto, shadowed_inputs: list[str]
) -> None:
    """Evaluate statically-known foldable nodes with onnx.reference; inline results.

    Inputs shadowed by initializers are appended to *shadowed_inputs* instead of
    warned about here: this function runs at a recursion depth that varies with
    subgraph nesting, so a fixed ``stacklevel`` could never attribute the
    warning to the user's frame. fold_constants() emits the aggregate warning."""
    from onnx.reference import ReferenceEvaluator

    # Initializers stay as protos; arrays materialize lazily (and only below
    # the size cap) so folding never duplicates the full weight set in memory.
    init_protos: dict[str, onnx.TensorProto] = {i.name: i for i in g.initializer}
    init_cache: dict[str, np.ndarray] = {}

    def _initializer_nbytes(t: onnx.TensorProto) -> int:
        if t.raw_data:
            return len(t.raw_data)
        itemsize = onnx.helper.tensor_dtype_to_np_dtype(t.data_type).itemsize
        return int(np.prod(t.dims)) * itemsize if t.dims else itemsize

    def _initializer_array(name: str) -> np.ndarray:
        arr = init_cache.get(name)
        if arr is None:
            arr = init_cache[name] = numpy_helper.to_array(init_protos[name])
        return arr

    shadowed_inputs.extend({vi.name for vi in g.input} & init_protos.keys())
    static_shapes: dict[str, tuple[int, ...]] = {}
    for vi in list(g.input) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.shape.dim and all(d.HasField("dim_value") for d in tt.shape.dim):
            static_shapes[vi.name] = tuple(d.dim_value for d in tt.shape.dim)

    graph_outputs = {o.name for o in g.output}
    # Consumptions left to process per value: folded intermediates are dropped
    # (or flushed to initializers) after their last consumer, so long constant
    # chains do not retain every intermediate until the end of the pass.
    remaining: dict[str, int] = {}
    for node in g.node:
        for name in (*node.input, *iter_subgraph_inputs(node)):
            remaining[name] = remaining.get(name, 0) + 1

    folded: dict[str, np.ndarray] = {}
    kept: list[onnx.NodeProto] = []
    used_by_kept: set[str] = set()
    new_initializers: list[tuple[str, np.ndarray]] = []

    def _try_fold(node: onnx.NodeProto) -> dict[str, np.ndarray] | None:
        if node.op_type == "Constant":
            attr = node.attribute[0]
            if attr.name == "value":
                # C4: tensor-valued Constant — fast path
                return {node.output[0]: numpy_helper.to_array(attr.t)}
            # C4: non-tensor Constant (value_float, value_int, etc.) — fall
            # through to the ReferenceEvaluator path below; "Constant" is in
            # _FOLDABLE and node.input is empty so the all-inputs-known check
            # passes vacuously.
        elif (
            node.op_type == "Shape"
            and node.input[0] in static_shapes
            and not node.attribute
        ):
            return {
                node.output[0]: np.array(static_shapes[node.input[0]], dtype=np.int64)
            }
        if node.op_type not in _FOLDABLE:
            return None
        if not all(i in init_protos or i in folded for i in node.input):
            return None
        # Size-cap inputs before materializing anything (folded values were
        # already capped when produced).
        if any(
            i not in folded and _initializer_nbytes(init_protos[i]) > _FOLD_SIZE_LIMIT
            for i in node.input
        ):
            return None
        feeds = {
            i: folded[i] if i in folded else _initializer_array(i) for i in node.input
        }
        node_model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [node],
                "fold",
                [],
                [onnx.helper.make_empty_tensor_value_info(o) for o in node.output],
            ),
            opset_imports=list(model.opset_import),
            ir_version=model.ir_version,
        )
        try:
            results = ReferenceEvaluator(node_model).run(None, feeds)
        except Exception:
            return None
        out = {
            name: np.asarray(val)
            for name, val in zip(node.output, results, strict=True)
        }
        if any(a.nbytes > _FOLD_SIZE_LIMIT for a in out.values()):
            return None
        return out

    for node in g.node:
        results = _try_fold(node)
        if results is None:
            kept.append(node)
            # C2: subgraph captures count as uses so that values folded in the
            # main graph but consumed only inside subgraphs are kept.
            used_by_kept.update(node.input)
            used_by_kept.update(iter_subgraph_inputs(node))
        else:
            folded.update(results)
        for name in (*node.input, *iter_subgraph_inputs(node)):
            remaining[name] -= 1
            if remaining[name] == 0:
                init_cache.pop(name, None)
                if name in folded:
                    arr = folded.pop(name)
                    if name in used_by_kept or name in graph_outputs:
                        new_initializers.append((name, arr))

    del g.node[:]
    g.node.extend(kept)
    for name, arr in new_initializers:
        g.initializer.append(numpy_helper.from_array(arr, name=name))
    for name, arr in folded.items():  # never-consumed folds (e.g. graph outputs)
        if name in graph_outputs:
            g.initializer.append(numpy_helper.from_array(arr, name=name))


def fold_constants(model: onnx.ModelProto) -> onnx.ModelProto:
    shadowed: list[str] = []
    apply_graphs_topdown(
        model.graph, lambda g: _fold_constants_graph(model, g, shadowed)
    )
    if shadowed:
        names = sorted(set(shadowed))
        preview = ", ".join(names[:5]) + (", ..." if len(names) > 5 else "")
        warnings.warn(
            f"{len(names)} initializer(s) shadow graph inputs ({preview}); the "
            "ONNX spec treats these as caller-overridable defaults, but they are "
            "baked in as constants and removed from the converted signature",
            stacklevel=3,  # I1: points at preprocess()'s caller, not fold_constants
        )
    return model
