# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for MatMul, Gemm, and Einsum lowerings (Task 10)."""

import zlib

import numpy as np
import pytest

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import (
    assert_parity,
    requires_coreai_runtime,
    single_op_model,
    skip_on_compute_unit,
)

pytestmark = [pytest.mark.ops, requires_coreai_runtime]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# MatMul  (numpy semantics: 2-D, batched N-D with broadcast, 1-D edge cases)
# ---------------------------------------------------------------------------

MATMUL_SHAPES = [
    ((2, 3), (3, 4)),  # plain 2-D
    ((5, 2, 3), (5, 3, 4)),  # batched 3-D
    ((1, 2, 3), (7, 3, 4)),  # broadcast batch dim
    ((7, 2, 3), (3, 4)),  # rank mismatch: 3-D x 2-D
    ((3,), (3, 4)),  # 1-D x 2-D
    ((2, 3), (3,)),  # 2-D x 1-D
    pytest.param(
        (3,),
        (3,),
        marks=skip_on_compute_unit(
            "cpu",
            "cpu_only",
            reason="CPU specializer fails to load models with a rank-0 "
            "(scalar) output — upstream issue; default/GPU/ANE are fine",
        ),
        id="dot-1d-1d",
    ),  # 1-D x 1-D (dot, rank-0 result)
]


@pytest.mark.parametrize(("a_shape", "b_shape"), MATMUL_SHAPES)
async def test_matmul(a_shape, b_shape):
    rng = np.random.default_rng(_seed(f"matmul-{a_shape}-{b_shape}"))
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    model = single_op_model("MatMul", {"a": a, "b": b})
    await assert_parity(model, {"a": a, "b": b}, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Integer operands  (valid ONNX since MatMul-9/Gemm-13/Einsum-12).
# aicode.broadcasting_batch_matmul only accepts float/complex tensors (its
# verifier failure aborts the whole host process at .aimodel load time), so
# MatMul evaluates integer operands in float32 and casts the result back;
# Gemm/Einsum still reject integers (no observed models need them).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((2, 3), (3, 4)), ((5, 2, 3), (3, 4))],
)
async def test_matmul_integer(dtype, a_shape, b_shape):
    # Small index/geometry-style integer matmul (e.g. BEVDet's view transform).
    rng = np.random.default_rng(_seed(f"matmul-int-{dtype}-{a_shape}"))
    a = rng.integers(-8, 8, a_shape).astype(dtype)
    b = rng.integers(-8, 8, b_shape).astype(dtype)
    model = single_op_model("MatMul", {"a": a, "b": b})
    await assert_parity(model, {"a": a, "b": b})


@pytest.mark.parametrize("op", ["Gemm", "Einsum"])
def test_integer_matmul_rejected(op):
    a = np.ones((2, 3), dtype=np.int32)
    b = np.ones((3, 4), dtype=np.int32)
    attrs = {"equation": "ij,jk->ik"} if op == "Einsum" else None
    model = single_op_model(op, {"a": a, "b": b}, attrs=attrs)
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="floating-point"):
        converter.to_coreai()


# ---------------------------------------------------------------------------
# Gemm  (alpha * A' @ B' + beta * C; transA/transB; optional C)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trans_a", [0, 1])
@pytest.mark.parametrize("trans_b", [0, 1])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("beta", [1.0, 0.0, 2.0])
async def test_gemm(trans_a, trans_b, alpha, beta):
    rng = np.random.default_rng(_seed(f"gemm-{trans_a}-{trans_b}-{alpha}-{beta}"))
    a_shape = (3, 2) if trans_a else (2, 3)
    b_shape = (4, 3) if trans_b else (3, 4)
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    c = rng.standard_normal((4,)).astype(np.float32)
    model = single_op_model(
        "Gemm",
        {"a": a, "b": b, "c": c},
        attrs={"alpha": alpha, "beta": beta, "transA": trans_a, "transB": trans_b},
    )
    await assert_parity(model, {"a": a, "b": b, "c": c}, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("c_shape", [(2, 4), ()], ids=["full", "scalar"])
async def test_gemm_c_broadcast_shapes(c_shape):
    rng = np.random.default_rng(_seed(f"gemm-c-{c_shape}"))
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    c = rng.standard_normal(c_shape).astype(np.float32)
    model = single_op_model(
        "Gemm", {"a": a, "b": b, "c": c}, attrs={"alpha": 0.5, "beta": 2.0}
    )
    await assert_parity(model, {"a": a, "b": b, "c": c}, rtol=1e-3, atol=1e-3)


async def test_gemm_no_c():
    rng = np.random.default_rng(_seed("gemm-no-c"))
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    model = single_op_model("Gemm", {"a": a, "b": b}, attrs={"alpha": 0.5})
    await assert_parity(model, {"a": a, "b": b}, rtol=1e-3, atol=1e-3)


async def test_gemm_const_weight_transb_transposed_at_convert():
    # torch exports every nn.Linear as Gemm(x, W, b, transB=1) with W an
    # initializer.  The runtime pipeline never folds transpose-of-constant, so
    # the weight must be transposed at conversion time — no coreai.transpose
    # may survive in the converted program.
    rng = np.random.default_rng(_seed("gemm-transb-const"))
    a = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((16, 8)).astype(np.float32)
    c = rng.standard_normal((16,)).astype(np.float32)
    model = single_op_model(
        "Gemm", {"a": a}, attrs={"transB": 1}, initializers={"w": w, "c": c}
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    assert "coreai.transpose" not in str(converter.to_coreai())
    await assert_parity(model, {"a": a}, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# float16  (lowering is dtype-generic; loading is a known runtime limitation)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Core AI runtime fails to load float16 matmul programs (Program load "
        "failure 0x10004) — runtime limitation, not a lowering bug; same "
        "tracked Core AI f16 issue as tests/test_ops_unary.py"
    )
)
@pytest.mark.parametrize("op", ["MatMul", "Gemm"])
async def test_f16_matmul_gemm(op):
    rng = np.random.default_rng(_seed(f"f16-{op}"))
    a = rng.standard_normal((2, 3)).astype(np.float16)
    b = rng.standard_normal((3, 4)).astype(np.float16)
    model = single_op_model(op, {"a": a, "b": b})
    await assert_parity(model, {"a": a, "b": b}, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Einsum  (closed set: permutations + two-operand bmm-reducible contractions)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# com.microsoft::Inverse  (batched analytic inverse for trailing 1x1/2x2/3x3;
# seen in FOMM's dense-motion network as a batch of 2x2 affine jacobians)
# ---------------------------------------------------------------------------


def _inverse_model(x: np.ndarray):
    from onnx import TensorProto, helper

    node = helper.make_node("Inverse", ["x"], ["out0"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "test_inverse",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, x.shape)],
        [helper.make_tensor_value_info("out0", TensorProto.FLOAT, x.shape)],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 21),
            helper.make_opsetid("com.microsoft", 1),
        ],
        ir_version=10,
    )


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (1, 10, 2, 2), (3, 3), (4, 3, 3), (1, 1), (5, 1, 1)],
    ids=["2x2", "batched-2x2", "3x3", "batched-3x3", "1x1", "batched-1x1"],
)
async def test_inverse(shape):
    rng = np.random.default_rng(_seed(f"inverse-{shape}"))
    n = shape[-1]
    # Diagonal dominance keeps every matrix in the batch well-conditioned.
    x = (rng.standard_normal(shape) + 4 * np.eye(n)).astype(np.float32)
    model = _inverse_model(x)
    await assert_parity(model, {"x": x}, rtol=1e-3, atol=1e-3)


def test_inverse_large_matrix_rejected():
    x = np.eye(4, dtype=np.float32)[None]
    model = _inverse_model(x)
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match="3x3"):
        converter.to_coreai()


EINSUM_TWO_OPERAND_CASES = [
    ("ij,jk->ik", (2, 3), (3, 4)),
    ("bij,bjk->bik", (5, 2, 3), (5, 3, 4)),
    ("bhqd,bhkd->bhqk", (2, 3, 4, 5), (2, 3, 6, 5)),
    ("bhqk,bhkd->bhqd", (2, 3, 4, 6), (2, 3, 6, 5)),
    ("ij,jk", (2, 3), (3, 4)),  # implicit output -> ik
]


@pytest.mark.parametrize(("equation", "a_shape", "b_shape"), EINSUM_TWO_OPERAND_CASES)
async def test_einsum_two_operands(equation, a_shape, b_shape):
    rng = np.random.default_rng(_seed(f"einsum-{equation}"))
    a = rng.standard_normal(a_shape).astype(np.float32)
    b = rng.standard_normal(b_shape).astype(np.float32)
    model = single_op_model("Einsum", {"a": a, "b": b}, attrs={"equation": equation})
    await assert_parity(model, {"a": a, "b": b}, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("equation", ["ij->ji", "ij->ij"])
async def test_einsum_single_operand(equation):
    rng = np.random.default_rng(_seed(f"einsum-{equation}"))
    a = rng.standard_normal((2, 3)).astype(np.float32)
    model = single_op_model("Einsum", {"a": a}, attrs={"equation": equation})
    await assert_parity(model, {"a": a}, rtol=1e-3, atol=1e-3)


def test_einsum_diagonal_rejected():
    # Repeated label within one operand (diagonal extraction) is unsupported.
    a = np.zeros((3, 3), dtype=np.float32)
    model = single_op_model("Einsum", {"a": a}, attrs={"equation": "ii->i"})
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match=r"ii->i"):
        converter.to_coreai()
