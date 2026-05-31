"""Point-cloud preprocessing helpers."""

from __future__ import annotations

import numpy as np


def normalize_xyz(points: np.ndarray) -> np.ndarray:
    """Center and scale XYZ coordinates while preserving remaining channels."""
    if points.size == 0:
        return points.astype(np.float32)
    output = points.astype(np.float32).copy()
    xyz = output[:, :3]
    xyz -= xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 0:
        xyz /= scale
    output[:, :3] = xyz
    return output


def sample_points(points: np.ndarray, num_points: int, seed: int | None = None) -> np.ndarray:
    """Sample or pad a point cloud to a fixed number of points."""
    rng = np.random.default_rng(seed)
    if len(points) == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    replace = len(points) < num_points
    indices = rng.choice(len(points), size=num_points, replace=replace)
    return points[indices].astype(np.float32)

