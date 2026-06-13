# Copyright 2026 coreai-onnx contributors.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end parity tests on small torch-exported models."""

import io

import onnx
import pytest
import torch
from torch import nn

from .helpers import assert_parity, requires_coreai_runtime

pytestmark = [pytest.mark.e2e, pytest.mark.slow, requires_coreai_runtime]


def export_to_onnx(module: nn.Module, example: torch.Tensor) -> onnx.ModelProto:
    module.eval()
    buf = io.BytesIO()
    torch.onnx.export(module, (example,), buf, opset_version=18, dynamo=False)
    return onnx.load_from_string(buf.getvalue())


class ResNetBlockNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.relu(self.bn(self.conv(x)) + x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


class EncoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class DepthwiseCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU6(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8),
            nn.Conv2d(8, 16, 1),
            nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


async def _export_and_check(module: nn.Module, example: torch.Tensor) -> None:
    model = export_to_onnx(module, example)
    input_name = model.graph.input[0].name
    await assert_parity(model, {input_name: example.numpy()}, rtol=1e-2, atol=1e-3)


async def test_resnet_block() -> None:
    torch.manual_seed(0)
    await _export_and_check(ResNetBlockNet(), torch.randn(1, 3, 32, 32))


async def test_transformer_encoder() -> None:
    torch.manual_seed(1)
    await _export_and_check(EncoderBlock(), torch.randn(1, 16, 64))


async def test_depthwise_cnn() -> None:
    torch.manual_seed(2)
    await _export_and_check(DepthwiseCNN(), torch.randn(1, 3, 16, 16))
