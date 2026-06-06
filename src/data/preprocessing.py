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


def augment_point_cloud(
    points: np.ndarray,
    rotate: bool = True,
    jitter_std: float = 0.01,
    jitter_clip: float = 0.05,
    scale_range: tuple[float, float] = (0.8, 1.25),
    translation_range: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Apply train-time augmentation to a point cloud.

    Augmentations are PointNet-style and operate only on the XYZ channels:
    random rotation around the up (Z) axis, Gaussian per-point jitter,
    isotropic scaling and translation. Extra channels (e.g. RGB) are passed
    through unchanged.
    """
    if points.size == 0:
        return points.astype(np.float32)

    rng = np.random.default_rng(seed)
    output = points.astype(np.float32).copy()
    xyz = output[:, :3]

    if rotate:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.array(
            [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        xyz = xyz @ rotation.T

    if scale_range is not None:
        low, high = scale_range
        if high > low:
            xyz *= float(rng.uniform(low, high))

    if translation_range > 0.0:
        shift = rng.uniform(-translation_range, translation_range, size=(1, 3)).astype(np.float32)
        xyz += shift

    if jitter_std > 0.0:
        noise = rng.normal(loc=0.0, scale=jitter_std, size=xyz.shape).astype(np.float32)
        if jitter_clip > 0.0:
            noise = np.clip(noise, -jitter_clip, jitter_clip)
        xyz += noise

    output[:, :3] = xyz
    return output

