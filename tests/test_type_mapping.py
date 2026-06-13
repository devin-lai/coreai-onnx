# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import onnx
import pytest
from onnx import TensorProto

from coreai_onnx._type_mapping import (
    coreai_type_from_onnx_dtype,
    narrow_array,
    tensor_type_from_value_info,
)
from coreai_onnx.converter import Context


@pytest.fixture
def ctx():
    with Context() as c:
        yield c


@pytest.mark.usefixtures("ctx")
def test_basic_dtypes() -> None:
    from coreai._compiler.ir import F16Type, F32Type, IntegerType

    assert coreai_type_from_onnx_dtype(TensorProto.FLOAT) == F32Type.get()
    assert coreai_type_from_onnx_dtype(TensorProto.FLOAT16) == F16Type.get()
    assert coreai_type_from_onnx_dtype(TensorProto.BOOL) == IntegerType.get_signless(1)
    # narrowing policy
    assert coreai_type_from_onnx_dtype(TensorProto.INT64) == IntegerType.get_signed(32)
    assert coreai_type_from_onnx_dtype(TensorProto.DOUBLE) == F32Type.get()


@pytest.mark.usefixtures("ctx")
def test_unsupported_dtype_raises() -> None:
    with pytest.raises(ValueError, match="STRING"):
        coreai_type_from_onnx_dtype(TensorProto.STRING)


def test_narrow_array_int64() -> None:
    out = narrow_array(np.array([1, 2], dtype=np.int64), context="init 'w'")
    assert out.dtype == np.int32


def test_narrow_array_clamps_sentinels() -> None:
    # ONNX uses INT64_MAX/MIN as "to the end" sentinels (e.g. Slice ends)
    out = narrow_array(np.array([2**63 - 1, -(2**63)], dtype=np.int64), context="x")
    assert out.tolist() == [2**31 - 1, -(2**31)]


def test_narrow_array_clamps_negative_intmax_sentinel() -> None:
    # torch's ONNX exporter emits -(2**63 - 1) (= INT64_MIN + 1) as the
    # "unbounded negative" marker for Slice bounds (observed in the deepbox
    # model); it clamps to INT32_MIN like the other sentinels rather than
    # raising as an out-of-range value.
    out = narrow_array(np.array([-(2**63 - 1)], dtype=np.int64), context="x")
    assert out.tolist() == [-(2**31)]


def test_narrow_array_overflow_raises() -> None:
    with pytest.raises(OverflowError, match="init 'w'"):
        narrow_array(np.array([2**40], dtype=np.int64), context="init 'w'")


def test_narrow_array_passthrough() -> None:
    arr = np.array([1.0], dtype=np.float32)
    assert narrow_array(arr, context="x") is arr


def test_narrow_array_uint64() -> None:
    out = narrow_array(np.array([7], dtype=np.uint64), context="x")
    assert out.dtype == np.uint32
    assert out.tolist() == [7]
    with pytest.raises(OverflowError, match="uint64"):
        narrow_array(np.array([2**40], dtype=np.uint64), context="x")


def test_narrow_array_float64() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = narrow_array(np.array([1.5, 1e300], dtype=np.float64), context="x")
    assert out.dtype == np.float32
    assert out[0] == np.float32(1.5)
    assert out[1] == np.finfo(np.float32).max


@pytest.mark.usefixtures("ctx")
def test_tensor_type_static_and_dynamic() -> None:
    vi = onnx.helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "batch", 224])
    t = tensor_type_from_value_info(vi)
    assert t.shape[0] == 1
    assert t.is_dynamic_dim(1)
    assert t.shape[2] == 224
