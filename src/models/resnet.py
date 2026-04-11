from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Spoof(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
