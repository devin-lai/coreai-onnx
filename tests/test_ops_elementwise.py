# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import zlib

import numpy as np
import pytest

from .helpers import (
    assert_parity,
    requires_coreai_runtime,
    single_op_model,
    skip_on_compute_unit,
)

pytestmark = [pytest.mark.ops, requires_coreai_runtime]

BINARY_FLOAT = ["Add", "Sub", "Mul", "Div", "Pow", "Min", "Max"]
COMPARE = ["Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual"]
LOGICAL = ["And", "Or", "Xor"]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


@pytest.mark.parametrize("op", BINARY_FLOAT)
@pytest.mark.parametrize(
    "shapes", [((2, 3), (2, 3)), ((2, 3), (3,)), ((4, 1, 3), (2, 3))]
)
async def test_binary_float(op, shapes):
    rng = np.random.default_rng(_seed(f"{op}-{shapes}"))
    a = (rng.random(shapes[0]) + 0.5).astype(np.float32)
    b = (rng.random(shapes[1]) + 0.5).astype(np.float32)
    await assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b})


@pytest.mark.parametrize("op", COMPARE)
async def test_compare(op):
    rng = np.random.default_rng(_seed(f"compare-{op}"))
    a = rng.integers(0, 4, (3, 4)).astype(np.float32)
    b = rng.integers(0, 4, (3, 4)).astype(np.float32)
    await assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b})


@pytest.mark.parametrize("op", COMPARE)
async def test_compare_nan(op):
    """IEEE 754 / ONNX: every ordered comparison involving NaN is False."""
    a = np.array([np.nan, 1.0, np.nan, 2.0], dtype=np.float32)
    b = np.array([1.0, np.nan, np.nan, 2.0], dtype=np.float32)
    await assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b})


@pytest.mark.parametrize("op", ["Add", *COMPARE])
async def test_binary_f16(op):
    """float16 parity: comparisons on f16 graph inputs used to fail at
    .aimodel load (Program load failure 0x10004 / uncatchable MPSGraph abort);
    the lowerings now widen comparison operands to f32 (value-exact)."""
    if op == "Add":
        pytest.skip(
            "Core AI runtime fails to load float16 Add (Program load failure "
            "0x10004) — runtime limitation, not a lowering bug; unlike "
            "comparisons, f16 arithmetic has no value-exact f32 rewrite"
        )
    rng = np.random.default_rng(_seed(f"f16-{op}"))
    a = rng.integers(0, 4, (3, 4)).astype(np.float16)
    b = rng.integers(0, 4, (3, 4)).astype(np.float16)
    await assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b})


async def test_equal_bool_inputs():
    """Equal accepts tensor(bool) since opset 13.  Feeding bool graph inputs
    straight into the comparison crashed the host process at .aimodel load
    (SIGABRT, ANE Program load failure 0x10004); now decomposed via float32."""
    a = np.array([True, False, True, False])
    b = np.array([True, True, False, False])
    await assert_parity(single_op_model("Equal", {"a": a, "b": b}), {"a": a, "b": b})


@pytest.mark.parametrize("op", LOGICAL)
async def test_logical(op):
    rng = np.random.default_rng(_seed(f"logical-{op}"))
    a = rng.integers(0, 2, (3, 4)).astype(np.bool_)
    b = rng.integers(0, 2, (3, 4)).astype(np.bool_)
    await assert_parity(single_op_model(op, {"a": a, "b": b}), {"a": a, "b": b})


@pytest.mark.parametrize("dtype", [np.int32, np.int8, np.uint8])
async def test_bitwise_ops(dtype):
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
    b = np.array([3, 1, 7], dtype=dtype)
    if np.issubdtype(dtype, np.signedinteger):
        a = np.array([[-8, -1, 0], [1, 2, 7]], dtype=dtype)
        b = np.array([3, -2, 5], dtype=dtype)
    feed = {"a": a, "b": b}
    for op in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
        await assert_parity(single_op_model(op, feed), feed)
    await assert_parity(single_op_model("BitwiseNot", {"a": a}), {"a": a})


async def test_not():
    rng = np.random.default_rng(_seed("not"))
    a = rng.integers(0, 2, (3, 4)).astype(np.bool_)
    await assert_parity(single_op_model("Not", {"a": a}), {"a": a})


async def test_where():
    rng = np.random.default_rng(_seed("where"))
    c = rng.integers(0, 2, (2, 3)).astype(np.bool_)
    a = rng.random((2, 3)).astype(np.float32)
    b = rng.random((2, 3)).astype(np.float32)
    await assert_parity(
        single_op_model("Where", {"c": c, "a": a, "b": b}), {"c": c, "a": a, "b": b}
    )


async def test_mod_fmod():
    rng = np.random.default_rng(_seed("mod-fmod"))
    a = rng.random((2, 3)).astype(np.float32) * 10
    b = (rng.random((2, 3)) + 0.5).astype(np.float32)
    await assert_parity(
        single_op_model("Mod", {"a": a, "b": b}, attrs={"fmod": 1}), {"a": a, "b": b}
    )
    ai = rng.integers(1, 20, (2, 3)).astype(np.int32)
    bi = rng.integers(1, 5, (2, 3)).astype(np.int32)
    await assert_parity(single_op_model("Mod", {"a": ai, "b": bi}), {"a": ai, "b": bi})


async def test_mod_fmod0_negative_integers():
    a = np.array([-7, 7, -7, 7], dtype=np.int32)
    b = np.array([3, -3, -3, 3], dtype=np.int32)
    # expected (Python %): [2, -2, -1, 1]
    await assert_parity(single_op_model("Mod", {"a": a, "b": b}), {"a": a, "b": b})


@skip_on_compute_unit(
    "cpu",
    "cpu_only",
    reason="CPU fmod kernel is inexact for |x| >> |y| and returns NaN for "
    "finite % inf — upstream kernel divergence; default/GPU/ANE are exact",
)
async def test_mod_fmod1_large_magnitude_and_inf():
    """C fmod is exact: a decomposition through rounded float division returns
    results off by a whole |y| (fmod(1e8, 3) -> 0) and NaN for inf divisors."""
    a = np.array([1e8, 12345679.0, -5.5, 5.5, 7.0], dtype=np.float32)
    b = np.array([3.0, 7.0, 3.0, -3.0, np.inf], dtype=np.float32)
    # expected (C fmod): [1, 3, -2.5, 2.5, 7]
    await assert_parity(
        single_op_model("Mod", {"a": a, "b": b}, attrs={"fmod": 1}), {"a": a, "b": b}
    )


async def test_mod_fmod1_mixed_sign():
    """fmod=1 sign follows the dividend; cover negative operands (float + int)."""
    a = np.array([-7.5, 7.5, -7.5, 7.5, -1.0, -0.3], dtype=np.float32)
    b = np.array([2.0, -2.0, -2.0, 2.0, 3.0, 0.7], dtype=np.float32)
    await assert_parity(
        single_op_model("Mod", {"a": a, "b": b}, attrs={"fmod": 1}), {"a": a, "b": b}
    )
    ai = np.array([-7, 7, -7, 7], dtype=np.int32)
    bi = np.array([3, -3, -3, 3], dtype=np.int32)
    # expected (C fmod on ints): [-1, 1, -1, 1]
    await assert_parity(
        single_op_model("Mod", {"a": ai, "b": bi}, attrs={"fmod": 1}),
        {"a": ai, "b": bi},
    )


async def test_mod_fmod0_int_survives_optimize(tmp_path):
    """Integer Mod must survive AIProgram.optimize().

    The Core AI optimizer cancels the decomposition x - trunc_div(x, y) * y
    to 0 by real-number algebra (invalid for truncating integer division), so
    the lowering must produce the native modulo op. Regression: litehrnet's
    keypoint x-coordinate (ArgMax % 48) converted to all-zeros.
    """
    from coreai.runtime import NDArray

    import coreai_onnx

    from .helpers import _specialization_options

    a = np.array([0, 1, 47, 48, 49, 100, 1000, 3070], dtype=np.int32)
    b = np.array([48], dtype=np.int32)  # constant divisor, as in litehrnet
    program = coreai_onnx.convert(
        single_op_model("Mod", {"a": a}, initializers={"b": b})
    )
    program.optimize()
    asset = program.save_asset(tmp_path / "m.aimodel")
    async with asset.executable(_specialization_options()) as ai_model:
        fn = ai_model.load_function("main")
        out = await fn({"a": NDArray(a)})
    np.testing.assert_array_equal(np.asarray(out["out0"].numpy()), a % b)


async def test_div_int_truncates():
    a = np.array([-7, 7, -7, 7], dtype=np.int32)
    b = np.array([2, -2, -2, 2], dtype=np.int32)
    # ONNX Div on ints truncates toward zero: [-3, -3, 3, 3]
    await assert_parity(single_op_model("Div", {"a": a, "b": b}), {"a": a, "b": b})


async def test_min_max_variadic():
    rng = np.random.default_rng(_seed("min-max-variadic"))
    xs = {f"x{i}": (rng.random((2, 3)) + 0.1).astype(np.float32) for i in range(3)}
    await assert_parity(single_op_model("Min", xs), xs)
    await assert_parity(single_op_model("Max", xs), xs)


async def test_mean_variadic_broadcast():
    rng = np.random.default_rng(_seed("mean-variadic-broadcast"))
    feed = {
        "a": (rng.random((2, 3)) + 0.1).astype(np.float32),
        "b": (rng.random((3,)) + 0.1).astype(np.float32),
        "c": np.array(0.25, dtype=np.float32),
    }
    await assert_parity(single_op_model("Mean", feed), feed)


async def test_min_max_nan_propagation():
    """ONNX Min/Max follow np.minimum/maximum: NaN in either operand propagates."""
    a = np.array([np.nan, 1.0, 5.0], dtype=np.float32)
    b = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    await assert_parity(single_op_model("Min", {"a": a, "b": b}), {"a": a, "b": b})
    await assert_parity(single_op_model("Max", {"a": a, "b": b}), {"a": a, "b": b})


async def test_min_max_nan_propagation_three_operands():
    """A NaN in the FIRST operand must survive both pairwise reduction steps."""
    a = np.array([np.nan, 7.0], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    c = np.array([3.0, np.nan], dtype=np.float32)
    feed = {"a": a, "b": b, "c": c}
    await assert_parity(single_op_model("Min", feed), feed)
    await assert_parity(single_op_model("Max", feed), feed)


async def test_pow_int_base_float_exponent():
    """ONNX Pow: result has the base type; the exponent must not be truncated
    to int before exponentiation (4 ** 0.5 is 2, not 4 ** 0 == 1)."""
    a = np.array([4, 9, 16, 2], dtype=np.int32)
    b = np.array([0.5, 0.5, 0.5, 3.0], dtype=np.float32)
    await assert_parity(single_op_model("Pow", {"a": a, "b": b}), {"a": a, "b": b})


def test_variadic_reduce_missing_input_raises():
    """An input name absent from values_map is an upstream converter bug; the
    variadic lowering must raise instead of silently reducing over a subset."""
    from onnx import helper

    from coreai_onnx._lowerings._common import _variadic_reduce

    node = helper.make_node("Min", inputs=["a", "missing"], outputs=["out"])
    lower = _variadic_reduce(lambda a, b: a)
    with pytest.raises(KeyError):
        lower({"a": object()}, node, None)
