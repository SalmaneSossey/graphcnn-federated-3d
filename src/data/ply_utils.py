"""PLY loading utilities for point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_xyzrgb(path: str | Path) -> np.ndarray:
    """Read a PLY point cloud with XYZ and optional RGB columns.

    Returns an array shaped `[num_points, 6]`. Missing RGB values are filled
    with zeros. The `plyfile` package handles both ASCII and binary PLY. If it
    is unavailable, this falls back to a small ASCII-only reader.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        return read_ascii_ply_xyzrgb(path)

    path = Path(path)
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"PLY file is missing `{required}` vertex property: {path}")

    xyz = np.stack([vertex[name].astype(np.float32) for name in ("x", "y", "z")], axis=1)
    rgb_names = _resolve_rgb_names(names)
    if rgb_names is None:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    else:
        rgb = np.stack([vertex[name].astype(np.float32) for name in rgb_names], axis=1)
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def read_ascii_ply_xyzrgb(path: str | Path) -> np.ndarray:
    """Read a simple ASCII PLY file with XYZ and optional RGB columns.

    Returns an array shaped `[num_points, 6]`. Missing RGB values are filled
    with zeros. Prefer `read_ply_xyzrgb` for binary-capable loading.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header: list[str] = []
        for line in handle:
            header.append(line.strip())
            if line.strip() == "end_header":
                break

        if not header or "format ascii" not in "\n".join(header[:5]):
            raise ValueError(f"Only ASCII PLY is supported for now: {path}")

        property_names = [
            line.split()[-1]
            for line in header
            if line.startswith("property") and len(line.split()) >= 3
        ]
        rows = [line.split() for line in handle if line.strip()]

    if not rows:
        return np.empty((0, 6), dtype=np.float32)

    values = np.asarray(rows, dtype=np.float32)
    columns = {name: idx for idx, name in enumerate(property_names)}
    xyz = np.stack([values[:, columns[name]] for name in ("x", "y", "z")], axis=1)

    rgb_columns = _resolve_rgb_names(tuple(columns))
    if rgb_columns is not None:
        rgb = np.stack([values[:, columns[name]] for name in rgb_columns], axis=1)
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = np.zeros_like(xyz)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)


def _resolve_rgb_names(names: tuple[str, ...]) -> tuple[str, str, str] | None:
    """Return RGB property names from common PLY naming conventions."""
    for candidates in (("red", "green", "blue"), ("r", "g", "b")):
        if all(name in names for name in candidates):
            return candidates
    return None
