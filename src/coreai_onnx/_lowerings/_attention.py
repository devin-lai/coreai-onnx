# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowering for the ``coreai::ScaledDotProductAttention`` fused node.

The node is produced by ``coreai_onnx._fusion`` (and may also be
authored directly in a custom ONNX graph).  It is lowered to the same
``scaled_dot_product_attention`` composite op that ``coreai-torch`` emits for
``torch.nn.functional.scaled_dot_product_attention``: a private, non-inlined
``coreai.graph`` carrying a ``#coreai.composite_declaration`` attribute.  The
on-device compilers recognize the declaration and substitute their fused
attention kernel; the decomposition body below is the portable fallback.

Node contract (mirrors the composite's):
    inputs    query [.., L, E], key [.., S, E], value [.., S, E]
              (rank 3 or 4; rank 3 gets an implicit single head dim), plus an
              optional float additive mask broadcastable to [.., L, S]
    attribute scale (float) — defaults to E**-0.5 like torch SDPA
    output    [.., L, E]

Query/key/value must share their last dim E: the composite kernel requires
it.  Callers with a smaller query/key head dim (e.g. YOLO attention) must
zero-pad query and key up to E, which leaves the dot products unchanged.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    Context,
    DictAttr,
    F32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    Location,
    StringAttr,
    Value,
    tensor_type,
)
from .._ir import coreai_dialect as coreai
from .._utils import attrs, operand, operands

_COMPOSITE_NAME = "scaled_dot_product_attention"
_PERM_SWAP_LAST = np.array([0, 1, 3, 2], dtype=np.uint32)


def _materialize(x: Value) -> Value:
    """Force *x* into a freshly materialized buffer via a pad+slice identity.

    The GPU and Neural Engine specializers substitute their fused kernel for
    the SDPA composite, and that substitution misbehaves when an operand of
    the ``coreai.invoke`` is produced by certain op chains: a transpose
    producer yields silently wrong attention output, and e.g. a scalar-scaled
    ``Mul`` producer (torch MultiheadAttention exports) crashes the GPU
    compiler outright.  CPU execution is always correct, and the failing
    producer set is not cleanly characterizable, so every operand is routed
    through this barrier unconditionally (verified across CPU/GPU/ANE; see
    tests/test_fuse_attention.py).  Purely structural no-ops (same-type cast,
    same-shape reshape, zero pad, +0, 1-element concat, unit tile) are folded
    away before the kernel substitution and do not work as barriers.  Padding
    one zero column and slicing it back off survives — both are real data
    ops — and costs two copies of an attention input, which is negligible
    next to the attention itself.
    """
    rank = tensor_type(x).rank
    dims = list(tensor_type(x).shape)
    if any(d < 0 for d in dims):
        # Dynamic dims cannot be sliced back statically; silently skipping
        # the barrier would reintroduce the GPU/ANE failures.
        raise ValueError(
            "ScaledDotProductAttention: operands with dynamic dimensions "
            "cannot be safely passed to the fused kernel"
        )
    padding = np.zeros(2 * rank, dtype=np.uint32)
    padding[-1] = 1
    zero = coreai.constant(0, dtype=tensor_type(x).element_type)
    padded = coreai.pad(x, padding, zero, "constant")
    return coreai.slice_(padded, [0] * rank, dims, [1] * rank)


def _composite_decl(
    context: Context, input_names: list[str], scale: float
) -> Attribute:
    """The ``#coreai.composite_declaration`` attribute for one SDPA instance.

    Field-for-field identical to the declaration ``coreai-torch`` generates,
    so the OS compilers recognize it regardless of which frontend produced
    the model.
    """
    si64 = IntegerType.get_signed(64, context)

    def strings(names: list[str]) -> ArrayAttr:
        return ArrayAttr.get([StringAttr.get(n, context) for n in names], context)

    with Location.unknown(context):
        decl = DictAttr.get(
            {
                "input_names": strings(input_names),
                "output_names": strings(["output"]),
                "op_attrs": DictAttr.get(
                    {
                        "is_causal": BoolAttr.get(False, context),
                        "window_size": IntegerAttr.get(si64, 0),
                        "scale": FloatAttr.get(F32Type.get(context), scale),
                        "version": IntegerAttr.get(si64, 1),
                    },
                    context,
                ),
            },
            context,
        )
    return Attribute.parse(
        f'#coreai.composite_declaration<"{_COMPOSITE_NAME}" = {decl!s}>',
        context=context,
    )


def _sdpa_body(
    query: Value, key: Value, value: Value, attn_mask: Value | None, scale: float
) -> Value:
    """The composite's decomposition: softmax(scale*Q @ K^T [+ mask]) @ V."""
    ele_type = tensor_type(query).element_type
    scaled_q = coreai.broadcasting_mul(query, coreai.constant(scale, dtype=ele_type))
    key_t = coreai.transpose(key, _PERM_SWAP_LAST)
    scores = coreai.broadcasting_batch_matmul(scaled_q, key_t)
    if attn_mask is not None:
        if tensor_type(attn_mask).element_type != ele_type:
            attn_mask = coreai.cast(attn_mask, ele_type)
        scores = coreai.broadcasting_add(scores, attn_mask)
    weights = coreai.softmax(scores, tensor_type(scores).rank - 1)
    return coreai.broadcasting_batch_matmul(weights, value)


def replace_scaled_dot_product_attention(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    q, k, v = operands(values_map, node, [0, 1, 2])
    mask = operand(values_map, node, 3)

    rank = tensor_type(q).rank
    if rank not in (3, 4):
        raise ValueError(
            f"ScaledDotProductAttention: query rank must be 3 or 4, got {rank}"
        )
    if tensor_type(k).rank != rank or tensor_type(v).rank != rank:
        raise ValueError(
            "ScaledDotProductAttention: query, key and value must have equal rank"
        )
    head_dims = {
        tensor_type(x).shape[-1] for x in (q, k, v) if tensor_type(x).shape[-1] >= 0
    }
    if len(head_dims) > 1:
        raise ValueError(
            "ScaledDotProductAttention: query, key and value must share their "
            f"last dimension (got {sorted(head_dims)}); zero-pad query/key to "
            "the value head dim"
        )
    if (
        mask is not None
        and tensor_type(mask).element_type != tensor_type(q).element_type
    ):
        raise ValueError(
            "ScaledDotProductAttention: the mask must be a float additive mask "
            "with the query's element type"
        )

    scale = attrs(node).get("scale")
    if scale is None:
        head_dim = tensor_type(q).shape[-1]
        if head_dim < 0:
            raise ValueError(
                "ScaledDotProductAttention: the scale attribute is required "
                "when the head dim is dynamic"
            )
        scale = 1.0 / math.sqrt(float(head_dim))
    scale = float(scale)

    if rank == 3:
        # Insert an implicit single head dim; squeezed back after the call.
        q = coreai.expand_dims(q, [1])
        k = coreai.expand_dims(k, [1])
        v = coreai.expand_dims(v, [1])
        if mask is not None and tensor_type(mask).rank == 3:
            mask = coreai.expand_dims(mask, [1])

    # The GPU/ANE fused kernel misbehaves for various operand producers;
    # force every operand through a materialization barrier (see
    # _materialize).
    q, k, v = _materialize(q), _materialize(k), _materialize(v)
    if mask is not None:
        mask = _materialize(mask)

    input_names = ["query", "key", "value"]
    if mask is not None:
        input_names.append("attn_mask")
    decl = _composite_decl(q.context, input_names, scale)

    if mask is not None:

        @coreai.graph(private=True, no_inline=True, composite_decl=decl)
        def sdpa(q_: Value, k_: Value, v_: Value, m_: Value) -> Value:
            return _sdpa_body(q_, k_, v_, m_, scale)

        result = sdpa(q, k, v, mask)[0]
    else:

        @coreai.graph(private=True, no_inline=True, composite_decl=decl)
        def sdpa(q_: Value, k_: Value, v_: Value) -> Value:
            return _sdpa_body(q_, k_, v_, None, scale)

        result = sdpa(q, k, v)[0]

    if rank == 3:
        result = coreai.shrink_dims(result, [1])
    return result


REGISTRY: dict[str, Callable[..., Any]] = {
    "coreai::ScaledDotProductAttention": replace_scaled_dot_product_attention,
}
