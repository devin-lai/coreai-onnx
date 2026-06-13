# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model-level passes: lite copies, shape inference, opset normalization."""

import warnings
from collections.abc import Iterator

import numpy as np
import onnx
from google.protobuf.message import Message
from onnx import helper, numpy_helper, shape_inference, version_converter

from .._utils import iter_graph_nodes, iter_node_subgraphs

BASELINE_OPSET = 17  # lowerings target opset semantics in the 17-22 window
MAX_KNOWN_OPSET = 22

# Initializer payloads above this size are replaced by typed graph inputs in
# the lightweight copy used for check_model/infer_shapes.  Small tensors stay
# inline: shape inference reads constant values of shape-bearing inputs
# (Reshape/Slice/...), which are always tiny.
_LITE_PAYLOAD_LIMIT = 1 << 16  # 64 KiB


def _copy_fields(dst: Message, src: Message, skip: str) -> None:
    for fd, val in src.ListFields():
        if fd.name == skip:
            continue
        if fd.is_repeated:
            getattr(dst, fd.name).extend(val)
        elif fd.type == fd.TYPE_MESSAGE:
            getattr(dst, fd.name).CopyFrom(val)
        else:
            setattr(dst, fd.name, val)


def _lite_copy(model: onnx.ModelProto) -> onnx.ModelProto:
    """*model* itself, or a copy of it without the big initializer payloads.

    checker.check_model and shape_inference.infer_shapes serialize their whole
    argument: protobuf hard-fails at 2 GiB, and (under upb) the first
    serialization of a weight-bearing proto permanently retains a multiple of
    the model size in RSS.  Neither pass needs weight bytes, so initializers
    above _LITE_PAYLOAD_LIMIT become typed graph inputs in the copy.  (Their
    payloads consequently escape checker validation — an acceptable trade for
    being able to process big models at all.)
    """
    if all(i.ByteSize() <= _LITE_PAYLOAD_LIMIT for i in model.graph.initializer):
        return model
    lite = onnx.ModelProto()
    _copy_fields(lite, model, skip="graph")
    _copy_fields(lite.graph, model.graph, skip="initializer")
    input_names = {vi.name for vi in model.graph.input}
    for init in model.graph.initializer:
        if init.ByteSize() <= _LITE_PAYLOAD_LIMIT:
            lite.graph.initializer.add().CopyFrom(init)
        elif init.name not in input_names:
            lite.graph.input.append(
                helper.make_tensor_value_info(
                    init.name, init.data_type, list(init.dims)
                )
            )
    return lite


def _infer_shapes_lite(model: onnx.ModelProto) -> onnx.ModelProto:
    """shape_inference.infer_shapes via _lite_copy, merging annotations back.

    Nodes are taken from the inferred copy because inference also annotates
    If/Loop subgraph value_info; inputs and initializers are untouched by
    inference, so the originals (with weight payloads) stay in place.
    """
    inferred = shape_inference.infer_shapes(_lite_copy(model))
    g, ig = model.graph, inferred.graph
    for field in ("node", "value_info", "output"):
        del getattr(g, field)[:]
        getattr(g, field).extend(getattr(ig, field))
    return model


def _default_opset(model: onnx.ModelProto) -> int:
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            return o.version
    return MAX_KNOWN_OPSET


def _is_default_group_norm(node: onnx.NodeProto) -> bool:
    return node.op_type == "GroupNormalization" and node.domain in ("", "ai.onnx")


def _iter_subgraph_nodes(graph: onnx.GraphProto) -> Iterator[onnx.NodeProto]:
    """Yield every node strictly inside If/Loop subgraphs of *graph*."""
    for n in graph.node:
        for sg in iter_node_subgraphs(n):
            yield from iter_graph_nodes(sg)


def _upgrade_deprecated_group_norm(model: onnx.ModelProto) -> onnx.ModelProto:
    """Upgrade a GroupNormalization-bearing opset 18-20 model to opset 21.

    GroupNormalization-18 (per-group scale/bias, shape [G]) was deprecated and
    redefined at opset 21 (per-channel, shape [C]); the checker hard-rejects
    the 18-20 form.  onnx 1.21's GroupNormalization 20->21 expansion adapter
    is shadowed by a no-op CompatibleAdapter (upstream bug), so convert_version
    alone would silently keep per-group params: expand them here.  When the
    channel count or the scale/bias initializers cannot be resolved statically
    the model is returned unchanged (and rejected by the checker, as before —
    per-group semantics must never reach the per-channel lowering).

    Limits: GroupNormalization inside If/Loop bodies is not expanded — the
    model is returned unchanged so the checker still rejects it loudly.
    convert_version serializes the full weight-bearing proto, so a >2GiB
    opset-18-20 GroupNorm model fails there (acceptable: _lite_copy cannot
    help a pass that must rewrite the real initializers).
    """
    if any(_is_default_group_norm(n) for n in _iter_subgraph_nodes(model.graph)):
        return model
    inferred = shape_inference.infer_shapes(_lite_copy(model))
    channels: dict[str, int] = {}
    for vi in list(inferred.graph.input) + list(inferred.graph.value_info):
        dims = vi.type.tensor_type.shape.dim
        if len(dims) >= 2 and dims[1].HasField("dim_value"):
            channels[vi.name] = dims[1].dim_value
    inits = {i.name: i for i in model.graph.initializer}
    expand: dict[str, int] = {}  # scale/bias initializer name -> channel count
    for node in model.graph.node:
        if node.op_type != "GroupNormalization" or node.domain not in ("", "ai.onnx"):
            continue
        c = channels.get(node.input[0])
        if c is None:
            return model
        # Per-group scale/bias have shape [num_groups], and the channel count
        # must split evenly into the groups. A model violating either is invalid
        # ONNX; leave it untouched so the checker rejects it, rather than
        # "repairing" mismatched params into a valid-looking opset-21 model (a
        # divisibility-only guard would let scale length 3 / num_groups 2 / C 6
        # through and expand bogus per-channel params).
        num_groups = next((a.i for a in node.attribute if a.name == "num_groups"), 0)
        if num_groups < 1 or c % num_groups:
            return model
        for name in node.input[1:3]:
            init = inits.get(name)
            if (
                init is None
                or len(init.dims) != 1
                or init.dims[0] != num_groups
                or expand.setdefault(name, c) != c
            ):
                return model
    model = version_converter.convert_version(model, 21)
    inits = {i.name: i for i in model.graph.initializer}
    for name, c in expand.items():
        arr = numpy_helper.to_array(inits[name])
        inits[name].CopyFrom(
            numpy_helper.from_array(np.repeat(arr, c // arr.shape[0]), name)
        )
    return model


def normalize_opset(model: onnx.ModelProto) -> onnx.ModelProto:
    v = _default_opset(model)
    if v < BASELINE_OPSET:
        model = version_converter.convert_version(model, BASELINE_OPSET)
    elif 18 <= v <= 20 and any(
        _is_default_group_norm(n) for n in iter_graph_nodes(model.graph)
    ):
        model = _upgrade_deprecated_group_norm(model)
    elif v > MAX_KNOWN_OPSET:
        warnings.warn(
            f"model opset {v} is newer than the supported window "
            f"({BASELINE_OPSET}-{MAX_KNOWN_OPSET}); attempting conversion as-is",
            stacklevel=3,  # I1: points at preprocess()'s caller, not normalize_opset
        )
    return model
