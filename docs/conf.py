# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coreai_onnx.__version__ import __version__

project = "coreai-onnx"
author = "coreai-onnx contributors"
release = __version__
version = __version__

extensions = ["myst_parser"]
html_theme = "shibuya"
exclude_patterns = ["_build", "superpowers"]

# Serve the repo-root llms.txt at the site root (llms.txt convention).
html_extra_path = ["../llms.txt"]
