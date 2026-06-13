# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lowering for the ONNX LSTM op.

The recurrence runs as a ``coreai.while`` loop (one per direction): the
input-to-gate projection ``X @ W^T + b`` has no timestep dependency and is
hoisted out of the loop as a single batched matmul; each iteration then adds
the recurrent projection ``H @ R^T``, applies the gate activations, and writes
the step's hidden state into the result buffer with ``slice_update``. This
keeps the emitted program size independent of sequence length (no unrolling).

Supported configuration — the window real exporters produce (PyTorch
``nn.LSTM`` and friends): layout 0, forward/reverse/bidirectional,
Sigmoid/Tanh/Relu activations, optional bias and initial states, full-length
sequences. Peepholes, ``clip``, ``input_forget``, per-batch ``sequence_lens``,
and parameterized activations are rejected with a clear error.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import onnx

from .._ir import FloatType, Location, Value, tensor_type
from .._ir import coreai_dialect as coreai
from .._utils import attrs, operand, operands
from ._common import _const_array

_GATE_ACTIVATIONS: dict[str, Callable[[Value], Value]] = {
    "Sigmoid": coreai.sigmoid,
    "Tanh": coreai.tanh,
    "Relu": coreai.relu,
}

_I32 = np.int32


def _static_shape(v: Value, what: str) -> list[int]:
    shape = list(tensor_type(v).shape)
    if any(d < 0 for d in shape):
        raise ValueError(f"LSTM: {what} must have a static shape, got {shape}")
    return shape


def _per_direction(v: Value, d: int) -> Value:
    """v[d] for a [num_directions, ...] operand, folded for constants."""
    arr = _const_array(v)
    if arr is not None:
        return coreai.constant(np.ascontiguousarray(arr[d]))
    shape = _static_shape(v, "operand")
    r = len(shape)
    starts = [d] + [0] * (r - 1)
    ends = [d + 1, *shape[1:]]
    return coreai.shrink_dims(coreai.slice_(v, starts, ends, [1] * r), [0])


def _transposed(v: Value) -> Value:
    """transpose(v) for a 2-D weight, folded at conversion time for constants
    (the runtime pipeline never folds transpose-of-constant; emitting it would
    re-copy the weight on every inference — same policy as the Gemm lowering)."""
    arr = _const_array(v)
    if arr is not None:
        return coreai.constant(np.ascontiguousarray(arr.T))
    return coreai.transpose(v, np.array([1, 0], dtype=np.uint32))


def _validate_lstm_attrs(node_attrs: dict[str, Any], num_directions: int) -> list[str]:
    """Reject unsupported attribute combinations; return per-direction
    activation names as a flat [f, g, h] * num_directions list."""
    if node_attrs.get("clip") is not None:
        raise ValueError("LSTM: the 'clip' attribute is not supported")
    if node_attrs.get("layout", 0) != 0:
        raise ValueError("LSTM: only layout=0 (sequence-major) is supported")
    if node_attrs.get("input_forget", 0) != 0:
        raise ValueError("LSTM: input_forget=1 (coupled gates) is not supported")
    if node_attrs.get("activation_alpha") or node_attrs.get("activation_beta"):
        raise ValueError(
            "LSTM: parameterized activations (activation_alpha/activation_beta) "
            "are not supported"
        )
    activations = node_attrs.get("activations") or ["Sigmoid", "Tanh", "Tanh"] * (
        num_directions
    )
    if len(activations) != 3 * num_directions:
        raise ValueError(
            f"LSTM: expected {3 * num_directions} activations for "
            f"{num_directions} direction(s), got {len(activations)}"
        )
    for name in activations:
        if name not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"LSTM: activation '{name}' is not supported "
                f"(supported: {', '.join(sorted(_GATE_ACTIVATIONS))})"
            )
    return activations


def _check_sequence_lens(
    values_map: dict[str, Value], node: onnx.NodeProto, seq_len: int
) -> None:
    """Per-batch sequence lengths cannot be honored by the full-length loop;
    accept only a constant vector that says every sequence is full-length."""
    seq_lens = operand(values_map, node, 4)
    if seq_lens is None:
        return
    arr = _const_array(seq_lens)
    if arr is None or not np.all(arr == seq_len):
        raise ValueError(
            "LSTM: per-batch 'sequence_lens' is not supported (only a constant "
            f"vector equal to the full sequence length {seq_len} is accepted)"
        )


def _lstm_single_direction(
    xw: Value,
    r_t: Value,
    h0: Value,
    c0: Value,
    y0: Value,
    acts: list[Callable[[Value], Value]],
    *,
    seq_len: int,
    batch: int,
    hidden: int,
    reverse: bool,
) -> tuple[Value, Value, Value]:
    """Run one direction's recurrence; return (Y [S,B,H], H_last, C_last).

    ``xw`` is the hoisted input projection [S, B, 4H] (bias already added),
    ``r_t`` the transposed recurrence weight [H, 4H]. A reverse direction
    walks the mirrored input and mirrors its output back, so the loop body
    itself is direction-agnostic.
    """
    act_f, act_g, act_h = acts
    if reverse:
        xw = coreai.reverse(xw, np.array([0], dtype=_I32))

    t0 = coreai.constant(np.array(0, dtype=_I32))
    n = coreai.constant(np.array(seq_len, dtype=_I32))

    while_loop: Any = coreai.while_(
        results=[t0.type, h0.type, c0.type, y0.type],
        inits=[t0, h0, c0, y0],
    )
    results, (before, after) = while_loop
    with before:
        t, h, c, y = before.arguments
        coreai.condition(coreai.not_equal(t, n), t, h, c, y)
    with after:
        t, h, c, y = after.arguments
        t1 = coreai.expand_dims(t, [0])  # [] -> [1]
        one = coreai.constant(np.array([1], dtype=_I32))
        t1_next = coreai.add(t1, one)
        zeros2 = coreai.constant(np.array([0, 0], dtype=_I32))
        strides3 = coreai.constant(np.array([1, 1, 1], dtype=_I32))

        # gates = xw[t] + h @ R^T : [B, 4H]
        start = coreai.concat(0, [t1, zeros2])
        end_x = coreai.concat(
            0, [t1_next, coreai.constant(np.array([batch, 4 * hidden], dtype=_I32))]
        )
        # A runtime-start slice has a dynamic result type; reshape to the
        # statically known [B, 4H] so downstream types stay static.
        xw_t = coreai.reshape(
            coreai.slice_(xw, start, end_x, strides3), [batch, 4 * hidden]
        )
        gates = coreai.broadcasting_add(xw_t, coreai.broadcasting_batch_matmul(h, r_t))

        # ONNX gate order along the 4H axis: input, output, forget, cell.
        def gate(k: int) -> Value:
            return coreai.slice_(
                gates, [0, k * hidden], [batch, (k + 1) * hidden], [1, 1]
            )

        i_g = act_f(gate(0))
        o_g = act_f(gate(1))
        f_g = act_f(gate(2))
        g_g = act_g(gate(3))

        c_next = coreai.broadcasting_add(
            coreai.broadcasting_mul(f_g, c), coreai.broadcasting_mul(i_g, g_g)
        )
        h_next = coreai.broadcasting_mul(o_g, act_h(c_next))

        end_h = coreai.concat(
            0, [t1_next, coreai.constant(np.array([batch, hidden], dtype=_I32))]
        )
        y_next = coreai.slice_update(
            y, start, end_h, strides3, coreai.expand_dims(h_next, [0])
        )
        t_next = coreai.add(t, coreai.constant(np.array(1, dtype=_I32)))
        coreai.yield_([t_next, h_next, c_next, y_next])

    _, h_last, c_last, y_out = list(results)
    if reverse:
        y_out = coreai.reverse(y_out, np.array([0], dtype=_I32))
    return y_out, h_last, c_last


def replace_lstm(
    values_map: dict[str, Value], node: onnx.NodeProto, loc: Location
) -> list[Value]:
    x, w, r = operands(values_map, node, [0, 1, 2])
    if not isinstance(tensor_type(x).element_type, FloatType):
        raise ValueError(
            f"LSTM: only floating-point inputs are supported, got element "
            f"type {tensor_type(x).element_type}"
        )
    if operand(values_map, node, 7) is not None:
        raise ValueError("LSTM: peephole weights (input P) are not supported")

    node_attrs = attrs(node)
    direction = node_attrs.get("direction", "forward")
    if direction not in ("forward", "reverse", "bidirectional"):
        raise ValueError(f"LSTM: unknown direction '{direction}'")
    num_directions = 2 if direction == "bidirectional" else 1

    activations = _validate_lstm_attrs(node_attrs, num_directions)

    seq_len, batch, _ = _static_shape(x, "input X")
    w_shape = _static_shape(w, "weight W")
    hidden = node_attrs.get("hidden_size", w_shape[1] // 4)
    if w_shape[0] != num_directions or w_shape[1] != 4 * hidden:
        raise ValueError(
            f"LSTM: weight W has shape {w_shape}, expected "
            f"[{num_directions}, {4 * hidden}, input_size]"
        )
    _check_sequence_lens(values_map, node, seq_len)

    b = operand(values_map, node, 3)
    initial_h = operand(values_map, node, 5)
    initial_c = operand(values_map, node, 6)

    ys: list[Value] = []
    h_lasts: list[Value] = []
    c_lasts: list[Value] = []
    for d in range(num_directions):
        w_d = _transposed(_per_direction(w, d))  # [I, 4H]
        r_d = _transposed(_per_direction(r, d))  # [H, 4H]
        xw = coreai.broadcasting_batch_matmul(x, w_d)  # [S, B, 4H]
        if b is not None:
            b_d = _per_direction(b, d)  # [8H] = Wb ++ Rb
            wb = coreai.slice_(b_d, [0], [4 * hidden], [1])
            rb = coreai.slice_(b_d, [4 * hidden], [8 * hidden], [1])
            xw = coreai.broadcasting_add(xw, coreai.broadcasting_add(wb, rb))

        # MPSGraph's compiler aborts the process ('MLIR pass manager failed')
        # when a while-loop's float loop-carried state is initialized from
        # literal zero constants in multi-loop programs (upstream Core AI /
        # MPSGraph bug, observed on bidirectional LSTMs). Derive the zero
        # buffers from xw so every float init is data-dependent, and route
        # constant initial states through the same barrier.
        zero_y = coreai.broadcasting_mul(
            coreai.slice_(xw, [0, 0, 0], [seq_len, batch, hidden], [1, 1, 1]),
            coreai.constant(0, dtype=tensor_type(x).element_type),
        )  # [S, B, H]
        zero_state = coreai.reshape(
            coreai.slice_(zero_y, [0, 0, 0], [1, batch, hidden], [1, 1, 1]),
            [batch, hidden],
        )  # [B, H]

        def initial_state(
            init: Value | None, direction_index: int = d, zero: Value = zero_state
        ) -> Value:
            if init is None:
                return zero
            init_d = _per_direction(init, direction_index)
            if _const_array(init) is None:
                return init_d
            return coreai.broadcasting_add(zero, init_d)

        acts = [_GATE_ACTIVATIONS[a] for a in activations[3 * d : 3 * d + 3]]
        y_d, h_last, c_last = _lstm_single_direction(
            xw,
            r_d,
            initial_state(initial_h),
            initial_state(initial_c),
            zero_y,
            acts,
            seq_len=seq_len,
            batch=batch,
            hidden=hidden,
            reverse=(direction == "reverse" or d == 1),
        )
        ys.append(coreai.expand_dims(y_d, [1]))  # [S, 1, B, H]
        h_lasts.append(coreai.expand_dims(h_last, [0]))  # [1, B, H]
        c_lasts.append(coreai.expand_dims(c_last, [0]))

    y = ys[0] if num_directions == 1 else coreai.concat(1, ys)
    y_h = h_lasts[0] if num_directions == 1 else coreai.concat(0, h_lasts)
    y_c = c_lasts[0] if num_directions == 1 else coreai.concat(0, c_lasts)

    # One value per non-empty output, in Y/Y_h/Y_c order (converter contract).
    all_outputs = [y, y_h, y_c]
    return [all_outputs[i] for i, name in enumerate(node.output) if name]


REGISTRY: dict[str, Callable[..., Any]] = {
    "LSTM": replace_lstm,
}
