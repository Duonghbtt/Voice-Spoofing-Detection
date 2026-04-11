from __future__ import annotations

import torch
import torch.nn as nn


class MFM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: int = 0,
        linear: bool = False,
    ) -> None:
        super().__init__()
        if linear:
            self.filter = nn.Linear(in_features, out_features * 2)
            self.channel_dim = -1
        else:
            if kernel_size is None:
                raise ValueError("kernel_size is required for convolutional MFM")
            self.filter = nn.Conv2d(
                in_features,
                out_features * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.channel_dim = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        first, second = torch.chunk(x, 2, dim=self.channel_dim)
        return torch.max(first, second)


class LCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            MFM(in_channels, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM(32, 64, kernel_size=1),
            MFM(64, 96, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM(96, 96, kernel_size=1),
            MFM(96, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            MFM(128, 128, kernel_size=1),
            MFM(128, 128, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            MFM(128 * 4 * 4, 256, linear=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
