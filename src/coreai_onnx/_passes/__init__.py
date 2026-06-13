# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model preprocessing: opset normalization, cleanup, constant folding.

preprocess() is the converter's entry; the individual passes are exposed
for tests and for targeted reuse.
"""

import onnx

from ._cleanup import eliminate_dead_nodes, prune_dead_initializers, remove_identity
from ._fold import fold_constants
from ._model import BASELINE_OPSET, _infer_shapes_lite, _lite_copy, normalize_opset

__all__ = [
    "BASELINE_OPSET",
    "eliminate_dead_nodes",
    "fold_constants",
    "normalize_opset",
    "preprocess",
    "prune_dead_initializers",
    "remove_identity",
]


def preprocess(model: onnx.ModelProto) -> onnx.ModelProto:
    # normalize_opset runs before the checker: deprecated GroupNormalization
    # (opset 18-20) must be upgraded before check_model rejects it.
    model = normalize_opset(model)
    onnx.checker.check_model(_lite_copy(model))
    model = _infer_shapes_lite(model)
    model = remove_identity(model)
    model = fold_constants(model)
    model = eliminate_dead_nodes(model)
    return prune_dead_initializers(model)
