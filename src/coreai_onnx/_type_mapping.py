# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""ONNX dtype/shape → Core AI IR type conversion and 64-bit narrowing."""

from collections.abc import Callable

import numpy as np
import onnx
from onnx import TensorProto

from ._ir import (
    BF16Type,
    F16Type,
    F32Type,
    Float8E4M3FNType,
    Float8E5M2Type,
    IntegerType,
    RankedTensorType,
    ShapedType,
    Type,
)

# Policy: Core AI narrows 64-bit; int64→si32, uint64→ui32, double→f32
# (same as coreai-torch's _NARROW_TORCH_DTYPE).
ONNX_TO_COREAI_DTYPE: dict[int, Callable[[], Type]] = {
    TensorProto.BOOL: lambda: IntegerType.get_signless(1),
    TensorProto.INT4: lambda: IntegerType.get_signed(4),
    TensorProto.UINT4: lambda: IntegerType.get_unsigned(4),
    TensorProto.INT8: lambda: IntegerType.get_signed(8),
    TensorProto.UINT8: lambda: IntegerType.get_unsigned(8),
    TensorProto.INT16: lambda: IntegerType.get_signed(16),
    TensorProto.UINT16: lambda: IntegerType.get_unsigned(16),
    TensorProto.INT32: lambda: IntegerType.get_signed(32),
    TensorProto.UINT32: lambda: IntegerType.get_unsigned(32),
    TensorProto.INT64: lambda: IntegerType.get_signed(32),  # narrowed
    TensorProto.UINT64: lambda: IntegerType.get_unsigned(32),  # narrowed
    TensorProto.FLOAT16: lambda: F16Type.get(),
    TensorProto.BFLOAT16: lambda: BF16Type.get(),
    TensorProto.FLOAT: lambda: F32Type.get(),
    TensorProto.DOUBLE: lambda: F32Type.get(),  # narrowed
    TensorProto.FLOAT8E4M3FN: lambda: Float8E4M3FNType.get(),
    TensorProto.FLOAT8E5M2: lambda: Float8E5M2Type.get(),
}

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)
# Keyed by np.dtype, not np.<type>: int64 arrays may carry the platform-distinct
# `longlong` scalar type (e.g. onnxruntime outputs), and `np.longlong is not
# np.int64` even though their dtypes compare and hash equal. A np.dtype lookup
# matches both; keying by `arr.dtype.type` would silently skip narrowing them.
_NARROW_NUMPY: dict[np.dtype, np.dtype] = {
    np.dtype(np.int64): np.dtype(np.int32),
    np.dtype(np.uint64): np.dtype(np.uint32),
    np.dtype(np.float64): np.dtype(np.float32),
}


def coreai_type_from_onnx_dtype(onnx_dtype: int) -> Type:
    factory = ONNX_TO_COREAI_DTYPE.get(onnx_dtype)
    if factory is None:
        name = TensorProto.DataType.Name(onnx_dtype)
        raise ValueError(f"ONNX dtype {name} is not supported by Core AI")
    return factory()


def narrow_array(arr: np.ndarray, *, context: str) -> np.ndarray:
    """Apply the 64→32-bit narrowing policy to a numpy array.

    INT64_MAX/MIN sentinel values (ONNX's "unbounded" markers, e.g. Slice
    ends) are clamped silently. Any other out-of-int32-range value raises.
    uint64 follows the same rules as int64, using INT64_MAX as the sentinel.
    float64 values are clipped to the float32 range and cast (no warning).
    """
    target = _NARROW_NUMPY.get(arr.dtype)
    if target is None:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(target)
        sentinel = arr == _INT64_MAX
        if np.issubdtype(arr.dtype, np.signedinteger):
            # INT64_MIN and -INT64_MAX (= INT64_MIN + 1) are both "unbounded
            # negative" markers: numpy's iinfo min, and the value torch's ONNX
            # exporter emits for open-ended Slice bounds (seen in deepbox).
            sentinel |= (arr == _INT64_MIN) | (arr == -_INT64_MAX)
        in_range = (arr >= info.min) & (arr <= info.max)
        if not np.all(in_range | sentinel):
            bad = arr[~(in_range | sentinel)].flat[0]
            raise OverflowError(
                f"{context}: {arr.dtype} value {bad} does not fit in {target} "
                "(Core AI narrows 64-bit integers to 32-bit)"
            )
        return np.clip(arr, info.min, info.max).astype(target)
    finfo = np.finfo(target)
    return np.clip(arr, finfo.min, finfo.max).astype(target)


def tensor_type_from_value_info(vi: onnx.ValueInfoProto) -> RankedTensorType:
    tt = vi.type.tensor_type
    elem = coreai_type_from_onnx_dtype(tt.elem_type)
    dims: list[int] = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:  # dim_param or unknown → dynamic
            dims.append(ShapedType.get_dynamic_size())
    return RankedTensorType.get(dims, elem)
