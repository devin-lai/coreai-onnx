# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


class CoreaiOnnxError(Exception):
    """Base for all coreai-onnx errors."""


class UnsupportedOpError(CoreaiOnnxError):
    """Raised before lowering when the model contains ops with no lowering.

    Aggregates ALL missing ops into one report.
    """

    def __init__(self, missing: dict[str, list[str]]) -> None:
        # missing: op key → example node names
        self.missing = missing
        lines = ["The following ONNX ops have no Core AI lowering:"]
        for key, nodes in sorted(missing.items()):
            examples = ", ".join(nodes[:3])
            lines.append(f"  {key} ({len(nodes)} node(s), e.g. {examples})")
        lines.append(
            "\nRegister a custom lowering to proceed:\n"
            "    @converter.register_onnx_lowering"
            f'("{next(iter(sorted(missing)))}")\n'
            "    def lower(values_map, node, loc): ...\n"
            "Run `coreai-onnx inspect <model>` for a full coverage report."
        )
        super().__init__("\n".join(lines))


class ConversionError(CoreaiOnnxError):
    """A lowering failed for a specific node."""

    def __init__(self, node_name: str, op_key: str, cause: Exception) -> None:
        self.node_name = node_name
        self.op_key = op_key
        super().__init__(f"Failed to lower node '{node_name}' (op {op_key}): {cause}")


class ModelValidationError(CoreaiOnnxError):
    """The input ONNX model failed to load or run on ONNX Runtime.

    Raised by the pre-conversion validation gate (``validate_onnxruntime``)
    before any conversion work begins, so a model that passes the static
    ``onnx.checker`` but cannot build an InferenceSession or errors at run time
    is reported as a problem with the input rather than surfacing later as an
    opaque conversion or runtime failure.
    """
