# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Generate docs/coverage.md from the live coreai_onnx._coverage module.

Usage:
    python tools/gen_coverage_table.py
"""

import sys
from pathlib import Path

# Allow running from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coreai_onnx._coverage import coverage_markdown, supported_ops

_DOCS_DIR = Path(__file__).parent.parent / "docs"
_OUT = _DOCS_DIR / "coverage.md"

_HEADER = """\
# Supported op lowerings

coreai-onnx currently supports **{n} built-in op lowerings**. The table below
lists every ONNX, Core AI composite, and internal fused op key that can be
converted without any extra code.

If your model uses ops not listed here, you can add support via a custom lowering
— see [Custom lowerings](custom-lowerings.md) for a worked example.

"""


def main() -> None:
    n = len(supported_ops())
    table = coverage_markdown()
    content = _HEADER.format(n=n) + table + "\n"
    _OUT.write_text(content)
    print(f"Written {_OUT}  ({n} lowerings)")


if __name__ == "__main__":
    main()
