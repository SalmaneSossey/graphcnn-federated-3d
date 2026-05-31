"""PLY loading utilities for point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ascii_ply_xyzrgb(path: str | Path) -> np.ndarray:
    """Read a simple ASCII PLY file with XYZ and optional RGB columns.

    Returns an array shaped `[num_points, 6]`. Missing RGB values are filled
    with zeros. TODO: add binary PLY support if the selected subset needs it.
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

    rgb_columns = ("red", "green", "blue")
    if all(name in columns for name in rgb_columns):
        rgb = np.stack([values[:, columns[name]] for name in rgb_columns], axis=1) / 255.0
    else:
        rgb = np.zeros_like(xyz)
    return np.concatenate([xyz, rgb], axis=1).astype(np.float32)

