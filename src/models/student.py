"""Compact student model for knowledge distillation."""

from __future__ import annotations

from torch import nn

from src.models.common import PointCloudMLP


class StudentPointCloudMLP(nn.Module):
    """Small student with substantially fewer parameters than default teachers."""

    def __init__(self, num_classes: int, input_channels: int = 6, hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = PointCloudMLP(
            num_classes=num_classes,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, points):
        """Return classification logits."""
        return self.backbone(points)

