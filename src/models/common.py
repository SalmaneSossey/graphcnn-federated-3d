"""Common model utilities and small point-cloud blocks."""

from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters."""
    params = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


class PointCloudMLP(nn.Module):
    """Simple baseline that embeds each point and pools globally."""

    def __init__(self, num_classes: int, input_channels: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Return logits for input shaped `[B, N, C]`."""
        if points.ndim != 3:
            raise ValueError(f"Expected [B, N, C] input, got shape {tuple(points.shape)}")
        features = self.point_encoder(points)
        pooled = features.max(dim=1).values
        return self.classifier(pooled)

