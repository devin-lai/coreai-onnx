# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Parity tests for the LSTM lowering (while-loop recurrence).

Reference configuration is the easyocr recognizer: bidirectional, defaults
activations, bias, explicit zero initial states, Y as the only output.
"""

import zlib

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import coreai_onnx
from coreai_onnx.errors import ConversionError

from .helpers import COREAI_RUNTIME_MARKS, assert_parity, requires_coreai_runtime

pytestmark = [pytest.mark.ops, *COREAI_RUNTIME_MARKS, requires_coreai_runtime]


def _seed(key: str) -> int:
    return zlib.crc32(key.encode()) & 0xFFFFFFFF


S, B, INPUT, H = 5, 2, 4, 3  # seq_len, batch, input_size, hidden_size


def _lstm_model(
    *,
    direction: str = "forward",
    with_bias: bool = True,
    with_initial_state: str | None = None,
    outputs: tuple[str, ...] = ("Y", "Y_h", "Y_c"),
    seed: str = "",
    extra_attrs: dict | None = None,
    seq_lens: np.ndarray | None = None,
    peepholes: bool = False,
) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    rng = np.random.default_rng(_seed(f"lstm-{direction}-{with_bias}-{seed}"))
    d = 2 if direction == "bidirectional" else 1
    x = rng.standard_normal((S, B, INPUT)).astype(np.float32)
    w = (rng.standard_normal((d, 4 * H, INPUT)) * 0.5).astype(np.float32)
    r = (rng.standard_normal((d, 4 * H, H)) * 0.5).astype(np.float32)

    inputs = {"x": x}
    initializers = {"w": w, "r": r}
    input_names = ["x", "w", "r"]

    if with_bias:
        initializers["b"] = (rng.standard_normal((d, 8 * H)) * 0.5).astype(np.float32)
        input_names.append("b")
    else:
        input_names.append("")

    if seq_lens is not None:
        initializers["seq_lens"] = seq_lens
        input_names.append("seq_lens")
    else:
        input_names.append("")

    if with_initial_state == "input":
        inputs["h0"] = rng.standard_normal((d, B, H)).astype(np.float32)
        inputs["c0"] = rng.standard_normal((d, B, H)).astype(np.float32)
        input_names.extend(["h0", "c0"])
    elif with_initial_state == "zero-const":
        # The easyocr recognizer shape: zero initial states as initializers.
        initializers["h0"] = np.zeros((d, B, H), dtype=np.float32)
        initializers["c0"] = np.zeros((d, B, H), dtype=np.float32)
        input_names.extend(["h0", "c0"])
    else:
        input_names.extend(["", ""])

    if peepholes:
        initializers["p"] = rng.standard_normal((d, 3 * H)).astype(np.float32)
        input_names.append("p")

    all_outputs = {"Y": (S, d, B, H), "Y_h": (d, B, H), "Y_c": (d, B, H)}
    output_names = [name if name in outputs else "" for name in all_outputs]
    node = helper.make_node(
        "LSTM",
        input_names,
        output_names,
        hidden_size=H,
        direction=direction,
        **(extra_attrs or {}),
    )
    graph = helper.make_graph(
        [node],
        "test_lstm",
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, a.shape)
            for n, a in inputs.items()
        ],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, all_outputs[n])
            for n in all_outputs
            if n in outputs
        ],
        initializer=[
            onnx.numpy_helper.from_array(a, name=n) for n, a in initializers.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
    )
    onnx.checker.check_model(model)
    return model, inputs


@pytest.mark.parametrize("direction", ["forward", "reverse", "bidirectional"])
async def test_lstm_directions_all_outputs(direction):
    model, inputs = _lstm_model(direction=direction)
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


async def test_lstm_no_bias():
    model, inputs = _lstm_model(with_bias=False, seed="nobias")
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("direction", ["forward", "bidirectional"])
@pytest.mark.parametrize("kind", ["input", "zero-const"])
async def test_lstm_initial_state(direction, kind):
    model, inputs = _lstm_model(
        direction=direction, with_initial_state=kind, seed="init"
    )
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


async def test_lstm_only_y_output():
    # The easyocr recognizer shape: Y requested, Y_h/Y_c omitted.
    model, inputs = _lstm_model(direction="bidirectional", outputs=("Y",), seed="y")
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


async def test_lstm_only_state_outputs():
    # Y omitted (empty first output), Y_h/Y_c requested.
    model, inputs = _lstm_model(outputs=("Y_h", "Y_c"), seed="state")
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


async def test_lstm_relu_activations():
    model, inputs = _lstm_model(
        seed="relu", extra_attrs={"activations": [b"Sigmoid", b"Relu", b"Relu"]}
    )
    await assert_parity(model, inputs, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"extra_attrs": {"clip": 1.5}}, "clip"),
        ({"extra_attrs": {"layout": 1}}, "layout"),
        ({"extra_attrs": {"input_forget": 1}}, "input_forget"),
        ({"extra_attrs": {"activations": [b"Sigmoid", b"Tanh", b"Affine"]}}, "Affine"),
        ({"seq_lens": np.array([5, 3], dtype=np.int32)}, "sequence_lens"),
        ({"peepholes": True}, "peephole"),
    ],
    ids=["clip", "layout", "input_forget", "activation", "seq_lens", "peepholes"],
)
def test_lstm_unsupported_configs_rejected(kwargs, match):
    model, _ = _lstm_model(seed="reject", **kwargs)
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    with pytest.raises(ConversionError, match=match):
        converter.to_coreai()


def test_lstm_padded_seq_lens_accepted():
    # sequence_lens that all equal seq_len carry no information; accept them.
    model, _inputs = _lstm_model(
        seed="fulllens", seq_lens=np.array([S, S], dtype=np.int32)
    )
    converter = coreai_onnx.OnnxConverter()
    converter.add_onnx_model(model)
    converter.to_coreai()
