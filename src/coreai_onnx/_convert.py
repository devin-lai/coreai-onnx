# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import onnx
from coreai.authoring import AIProgram

from .converter import OnnxConverter


def convert(
    model: onnx.ModelProto | str | Path,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    entrypoint_name: str = "main",
    repair: bool = False,
) -> AIProgram:
    """Convert an ONNX model to a Core AI AIProgram.

    Call .optimize() and .save_asset(path) on the result to produce a
    .aimodel bundle.

    With ``repair=True``, known-safe rewrites for documented Core AI runtime
    limitations are applied first (e.g. float16 programs are promoted to
    float32); each applied repair is reported via ``warnings.warn``. The
    rewrites are semantics-preserving — verify the result against ONNX Runtime
    (``verify``) to confirm parity.
    """
    if repair:
        from ._repair import apply_repairs

        if isinstance(model, str | Path):
            model = onnx.load(str(model))
        model, records = apply_repairs(model)
        for record in records:
            warnings.warn(f"coreai-onnx applied repair: {record.summary}", stacklevel=2)

    converter = OnnxConverter()
    converter.add_onnx_model(
        model,
        input_names=input_names,
        output_names=output_names,
        entrypoint_name=entrypoint_name,
    )
    return converter.to_coreai()
