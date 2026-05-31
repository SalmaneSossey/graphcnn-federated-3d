"""Vertical federated learning scaffold for XYZ/RGB feature ownership."""

from __future__ import annotations

import torch
from torch import nn


class VFLEntityEncoder(nn.Module):
    """Entity-side encoder that receives only one feature view."""

    def __init__(self, input_channels: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode entity-owned raw features into a pooled embedding."""
        features = self.encoder(points)
        return features.max(dim=1).values


class VFLServerClassifier(nn.Module):
    """Server classifier that sees only concatenated entity embeddings."""

    def __init__(self, embedding_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        """Classify using `[eA || eB]` without raw XYZ/RGB exchange."""
        return self.classifier(torch.cat([embedding_a, embedding_b], dim=1))


def split_xyz_rgb(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split `[B, N, 6]` point clouds into XYZ and RGB views."""
    if points.shape[-1] != 6:
        raise ValueError("VFL split expects XYZRGB input with 6 channels.")
    return points[..., :3], points[..., 3:]

