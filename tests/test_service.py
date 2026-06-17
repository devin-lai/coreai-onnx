# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for shared command helpers."""

from coreai_onnx import _service

_LICENSE = "BSD-3-Clause"
_REPOSITORY = "https://github.com/devin-lai/coreai-onnx"


def test_asset_metadata_identifies_converter():
    metadata = _service._asset_metadata()

    assert metadata.author == "coreai-onnx contributors"
    assert metadata.license == _LICENSE
    assert metadata.model_description == f"Converted with coreai-onnx: {_REPOSITORY}"
    assert metadata.creator_defined_metadata["coreai_onnx.repository"] == _REPOSITORY
    assert metadata.creator_defined_metadata["coreai_onnx.license"] == _LICENSE
