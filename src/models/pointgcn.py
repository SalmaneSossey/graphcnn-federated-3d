"""Lightweight PointGCN placeholder.

TODO: Replace this MLP-style scaffold with graph neighborhood construction,
edge feature aggregation, and PointGCN-specific message passing.
"""

from __future__ import annotations

from torch import nn

from src.models.common import PointCloudMLP


class PointGCN(nn.Module):
    """PointGCN-compatible interface accepting `[B, N, C]` point clouds."""

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

