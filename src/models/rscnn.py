"""Lightweight RS-CNN placeholder.

TODO: Replace this scaffold with relation-shape convolution blocks and local
region grouping once the dataset pipeline is ready.
"""

from __future__ import annotations

from torch import nn

from src.models.common import PointCloudMLP


class RSCNN(nn.Module):
    """RS-CNN-compatible interface accepting `[B, N, C]` point clouds."""

    def __init__(self, num_classes: int, input_channels: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = PointCloudMLP(
            num_classes=num_classes,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
        )

    def forward(self, points):
        """Return classification logits."""
        return self.backbone(points)

