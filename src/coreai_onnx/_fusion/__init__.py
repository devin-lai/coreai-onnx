# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph-fusion passes: attention (SDPA) and decomposed-activation rewrites."""

from ._activations import fuse_activations
from ._attention import fuse_attention

__all__ = ["fuse_activations", "fuse_attention"]
