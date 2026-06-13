# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowerings for matrix-multiplication ONNX ops (MatMul, Gemm, Einsum)."""

import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import FloatType, IntegerType, Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, operand, operands
from ._common import _const_array


def _require_float(x: Value, op_name: str) -> None:
    """Reject integer operands (valid ONNX): aicode.broadcasting_batch_matmul
    only accepts float/complex tensors, and its verifier failure is an
    uncatchable process abort at .aimodel load time — fail at conversion."""
    if not isinstance(tensor_type(x).element_type, FloatType):
        raise ValueError(
            f"{op_name}: only floating-point matrix multiplication is "
            f"supported by Core AI, got element type {tensor_type(x).element_type}"
        )


# ---------------------------------------------------------------------------
# MatMul  (numpy semantics: broadcasting_batch_matmul handles rank mismatch
# and batch-dim broadcasting; 1-D operands get a unit dim that is shrunk back)
# ---------------------------------------------------------------------------


def _matmul_values(a: Value, b: Value, op_name: str = "MatMul") -> Value:
    """Core AI matmul with ONNX/numpy rank-1 squeeze semantics.

    Core AI's batch_matmul accepts only floating-point operands, so integer
    MatMul (valid ONNX) is evaluated in float32 and cast back. float32 holds
    integers exactly up to 2**24; a product that accumulates beyond that would
    lose precision, which the small index/geometry-style integer matmuls seen
    in practice (e.g. BEVDet's view transform) stay well under.
    """
    elem = tensor_type(a).element_type
    integral = isinstance(elem, IntegerType) and elem.width > 1
    if integral:
        a = coreai.cast(a, np.float32)
        b = coreai.cast(b, np.float32)
    else:
        _require_float(a, op_name)
    ra, rb = tensor_type(a).rank, tensor_type(b).rank
    if ra < 1 or rb < 1:
        raise ValueError(f"{op_name}: operands must have rank >= 1")
    if ra == 1:
        a = coreai.expand_dims(a, [0])  # (k,) -> (1, k)
    if rb == 1:
        b = coreai.expand_dims(b, [1])  # (k,) -> (k, 1)
    out = coreai.broadcasting_batch_matmul(a, b)
    shrink = []
    if ra == 1:
        shrink.append(tensor_type(out).rank - 2)
    if rb == 1:
        shrink.append(tensor_type(out).rank - 1)
    if shrink:
        out = coreai.shrink_dims(out, shrink)
    if integral:
        out = coreai.cast(out, elem)
    return out


def replace_matmul(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    a, b = operands(values_map, node, [0, 1])
    return _matmul_values(a, b)


# ---------------------------------------------------------------------------
# Gemm  (Y = alpha * A' @ B' + beta * C)
# ---------------------------------------------------------------------------

_PERM_2D = np.array([1, 0], dtype=np.uint32)


def _transposed_2d(v: Value) -> Value:
    """transpose(v), folded at conversion time for constant operands — the
    runtime pipeline never folds transpose-of-constant, so emitting
    coreai.transpose would re-copy a constant weight on every inference."""
    arr = _const_array(v)
    if arr is not None:
        return coreai.constant(np.ascontiguousarray(arr.T))
    return coreai.transpose(v, _PERM_2D)


def replace_gemm(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    a, b = operands(values_map, node, [0, 1])
    _require_float(a, "Gemm")
    node_attrs = attrs(node)
    alpha = node_attrs.get("alpha", 1.0)
    beta = node_attrs.get("beta", 1.0)
    if node_attrs.get("transA", 0):
        a = _transposed_2d(a)
    if node_attrs.get("transB", 0):
        b = _transposed_2d(b)
    e_type = tensor_type(a).element_type
    out = coreai.broadcasting_batch_matmul(a, b)
    if alpha != 1.0:
        out = coreai.broadcasting_mul(out, coreai.constant(alpha, dtype=e_type))
    c = operand(values_map, node, 2)
    if c is None or beta == 0.0:
        return out
    if tensor_type(c).element_type != e_type:
        c = coreai.cast(c, e_type)
    if beta != 1.0:
        c = coreai.broadcasting_mul(c, coreai.constant(beta, dtype=e_type))
    return coreai.broadcasting_add(out, c)


# ---------------------------------------------------------------------------
# Einsum  (closed set: single-operand permutations and two-operand
# contractions reducible to transposes + broadcasting_batch_matmul)
# ---------------------------------------------------------------------------


def _parse_einsum_equation(equation: str, n_inputs: int) -> tuple[list[str], str]:
    """Split *equation* into per-operand label strings and the output labels."""
    eq = equation.replace(" ", "")
    if "." in eq:
        raise ValueError(f"Einsum: equation '{equation}': ellipsis is not supported")
    lhs, sep, out = eq.partition("->")
    terms = lhs.split(",")
    if len(terms) != n_inputs:
        raise ValueError(
            f"Einsum: equation '{equation}' names {len(terms)} operand(s) "
            f"but the node has {n_inputs} input(s)"
        )
    for t in terms:
        if not (t and t.isalpha()):
            raise ValueError(f"Einsum: equation '{equation}': invalid operand term")
        if len(set(t)) != len(t):
            raise ValueError(
                f"Einsum: equation '{equation}': repeated labels within one "
                "operand (diagonal/trace) are not supported"
            )
    counts = Counter(c for t in terms for c in t)
    if not sep:
        # Implicit output: labels appearing exactly once, sorted alphabetically.
        out = "".join(sorted(c for c, n in counts.items() if n == 1))
    elif len(set(out)) != len(out) or any(c not in counts for c in out):
        raise ValueError(f"Einsum: equation '{equation}': invalid output labels")
    return terms, out


def _maybe_transpose(x: Value, perm: list[int]) -> Value:
    if perm == list(range(len(perm))):
        return x
    return coreai.transpose(x, np.array(perm, dtype=np.uint32))


def _static_dims(x: Value, equation: str) -> list[int]:
    shape = list(tensor_type(x).shape)
    if any(d < 0 for d in shape):
        raise ValueError(
            f"Einsum: equation '{equation}' requires static operand shapes"
        )
    return shape


def _einsum_two_operands(
    equation: str, t1: str, t2: str, out: str, a: Value, b: Value
) -> Value:
    _require_float(a, "Einsum")
    s1, s2, s_out = set(t1), set(t2), set(out)
    batch = [c for c in t1 if c in s2 and c in s_out]
    contract = [c for c in t1 if c in s2 and c not in s_out]
    free1 = [c for c in t1 if c not in s2]
    free2 = [c for c in t2 if c not in s1]
    if any(c not in s_out for c in free1 + free2):
        raise ValueError(
            f"Einsum: equation '{equation}': summing over a label that appears "
            "in only one operand is not supported"
        )

    # Transpose to [batch..., free, contract] x [batch..., contract, free].
    a = _maybe_transpose(a, [t1.index(c) for c in batch + free1 + contract])
    b = _maybe_transpose(b, [t2.index(c) for c in batch + contract + free2])

    if len(free1) == 1 and len(contract) == 1 and len(free2) == 1:
        mm = coreai.broadcasting_batch_matmul(a, b)
    else:
        # Merge multi-label (or empty) free/contract groups into single dims.
        nb = len(batch)
        a_dims = _static_dims(a, equation)
        b_dims = _static_dims(b, equation)
        f1_dims, c_dims = a_dims[nb : nb + len(free1)], a_dims[nb + len(free1) :]
        f2_dims = b_dims[nb + len(contract) :]
        a = coreai.reshape(a, [*a_dims[:nb], math.prod(f1_dims), math.prod(c_dims)])
        b = coreai.reshape(b, [*b_dims[:nb], math.prod(c_dims), math.prod(f2_dims)])
        mm = coreai.broadcasting_batch_matmul(a, b)
        batch_dims = list(np.broadcast_shapes(tuple(a_dims[:nb]), tuple(b_dims[:nb])))
        mm = coreai.reshape(mm, batch_dims + f1_dims + f2_dims)

    inter = batch + free1 + free2  # label order of mm's axes
    return _maybe_transpose(mm, [inter.index(c) for c in out])


def replace_einsum(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    equation = attrs(node)["equation"]
    n_inputs = len([n for n in node.input if n])
    terms, out = _parse_einsum_equation(equation, n_inputs)
    vals = operands(values_map, node, list(range(n_inputs)))
    for t, v in zip(terms, vals, strict=True):
        if len(t) != tensor_type(v).rank:
            raise ValueError(
                f"Einsum: equation '{equation}': term '{t}' does not match "
                f"an operand of rank {tensor_type(v).rank}"
            )

    if n_inputs == 1:
        (x,) = vals
        (t,) = terms
        if sorted(t) != sorted(out):
            raise ValueError(
                f"Einsum: equation '{equation}': single-operand form must be a "
                "permutation (reductions are not supported)"
            )
        return _maybe_transpose(x, [t.index(c) for c in out])

    if n_inputs == 2:
        a, b = vals
        return _einsum_two_operands(equation, terms[0], terms[1], out, a, b)

    raise ValueError(
        f"Einsum: equation '{equation}': more than two operands are not supported"
    )


# ---------------------------------------------------------------------------
# com.microsoft::Inverse  (batched matrix inverse over the trailing two dims;
# closed-form adjugate/determinant for 1x1, 2x2, and 3x3 — the sizes ONNX
# exporters produce in practice, e.g. FOMM's batch of 2x2 affine jacobians)
# ---------------------------------------------------------------------------


def _mat_elem(x: Value, i: int, j: int, shape: list[int]) -> Value:
    """x[..., i:i+1, j:j+1] — a single matrix element, rank preserved."""
    r = len(shape)
    starts = [0] * (r - 2) + [i, j]
    ends = [*shape[:-2], i + 1, j + 1]
    return coreai.slice_(x, starts, ends, [1] * r)


def _neg(x: Value) -> Value:
    return coreai.broadcasting_mul(
        x, coreai.constant(-1, dtype=tensor_type(x).element_type)
    )


def _det2(a: Value, b: Value, c: Value, d: Value) -> Value:
    """Determinant of [[a, b], [c, d]] held as broadcast-compatible elements."""
    return coreai.broadcasting_sub(
        coreai.broadcasting_mul(a, d), coreai.broadcasting_mul(b, c)
    )


def replace_inverse(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> Value:
    (x,) = operands(values_map, node, [0])
    _require_float(x, "Inverse")
    shape = list(tensor_type(x).shape)
    if len(shape) < 2 or shape[-1] != shape[-2]:
        raise ValueError(
            f"Inverse: input must be square over the trailing two dims, got {shape}"
        )
    if any(d < 0 for d in shape):
        raise ValueError(f"Inverse: input must have a static shape, got {shape}")
    n = shape[-1]
    if n > 3:
        raise ValueError(
            f"Inverse: only matrices up to 3x3 are supported (closed-form "
            f"adjugate), got {n}x{n}"
        )
    rank = len(shape)

    if n == 1:
        return coreai.broadcasting_divide(
            coreai.constant(1, dtype=tensor_type(x).element_type), x
        )

    def elem(i: int, j: int) -> Value:
        return _mat_elem(x, i, j, shape)

    if n == 2:
        a, b = elem(0, 0), elem(0, 1)
        c, d = elem(1, 0), elem(1, 1)
        det = _det2(a, b, c, d)
        rows = [
            coreai.concat(rank - 1, [d, _neg(b)]),
            coreai.concat(rank - 1, [_neg(c), a]),
        ]
        adjugate = coreai.concat(rank - 2, rows)
        return coreai.broadcasting_divide(adjugate, det)

    # n == 3: inverse = adjugate / det, adjugate[i][j] = cofactor[j][i].
    m = [[elem(i, j) for j in range(3)] for i in range(3)]
    cof = [
        [
            _det2(
                m[(i + 1) % 3][(j + 1) % 3],
                m[(i + 1) % 3][(j + 2) % 3],
                m[(i + 2) % 3][(j + 1) % 3],
                m[(i + 2) % 3][(j + 2) % 3],
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    det = coreai.broadcasting_add(
        coreai.broadcasting_add(
            coreai.broadcasting_mul(m[0][0], cof[0][0]),
            coreai.broadcasting_mul(m[0][1], cof[0][1]),
        ),
        coreai.broadcasting_mul(m[0][2], cof[0][2]),
    )
    rows = [
        coreai.concat(rank - 1, [cof[0][i], cof[1][i], cof[2][i]]) for i in range(3)
    ]
    adjugate = coreai.concat(rank - 2, rows)
    return coreai.broadcasting_divide(adjugate, det)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Callable[..., Any]] = {
    "MatMul": replace_matmul,
    "Gemm": replace_gemm,
    "Einsum": replace_einsum,
    "com.microsoft::Inverse": replace_inverse,
}
