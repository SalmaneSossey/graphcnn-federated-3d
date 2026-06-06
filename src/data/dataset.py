"""Dataset definitions for ShapeNet point clouds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.ply_utils import read_ply_xyzrgb
from src.data.preprocessing import augment_point_cloud, normalize_xyz, sample_points


class ShapeNetPointCloudDataset(Dataset[tuple[torch.Tensor, int]]):
    """ShapeNet point-cloud dataset scaffold.

    Expected metadata path: `data/metadata/labeled_dataset.csv`.
    Expected point-cloud storage: `data/raw` or `data/cache`.
    """

    def __init__(
        self,
        metadata_csv: str | Path = "data/metadata/labeled_dataset.csv",
        pointcloud_root: str | Path = "data/raw",
        num_points: int = 1024,
        input_channels: int = 6,
        path_column: str | None = None,
        label_column: str | None = None,
        augment: bool = False,
        augment_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.pointcloud_root = Path(pointcloud_root)
        self.num_points = num_points
        self.input_channels = input_channels
        self.path_column = path_column
        self.label_column = label_column
        self.augment = augment
        self.augment_kwargs: dict[str, Any] = dict(augment_kwargs) if augment_kwargs else {}
        self.metadata = (
            pd.read_csv(self.metadata_csv) if self.metadata_csv.exists() else pd.DataFrame()
        )

        self.label_to_idx: dict[str, int] = {}

        if not self.metadata.empty:
            if self.path_column is None:
                self.path_column = self._infer_path_column(self.metadata)
            if self.label_column is None:
                self.label_column = self._infer_label_column(self.metadata)
            if self.label_column is not None and self.label_column != "label_idx":
                labels = sorted(self.metadata[self.label_column].astype(str).unique())
                self.label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.metadata.empty:
            raise IndexError("Dataset metadata is empty.")
        row = self.metadata.iloc[index]
        if self.path_column is None or self.path_column not in row:
            raise ValueError("Could not infer a point-cloud path column from metadata.")
        if self.label_column not in row:
            raise ValueError(f"Missing label column: {self.label_column}")

        point_path = Path(str(row[self.path_column]))
        if not point_path.is_absolute():
            point_path = self.pointcloud_root / point_path
        points = read_ply_xyzrgb(point_path)
        points = normalize_xyz(points)
        points = sample_points(points, self.num_points)
        if self.augment:
            points = augment_point_cloud(points, **self.augment_kwargs)
        if self.input_channels == 3:
            points = points[:, :3]
        elif self.input_channels != 6:
            raise ValueError("input_channels must be 3 or 6 for the current scaffold.")

        label_value = row[self.label_column]
        if self.label_column == "label_idx":
            label = int(label_value)
        else:
            label = self.label_to_idx[str(label_value)]
        return torch.from_numpy(points).float(), label

    @staticmethod
    def _infer_path_column(metadata: pd.DataFrame) -> str | None:
        candidates = (
            "file_path",
            "ply_path",
            "path",
            "filename",
            "file",
            "pointcloud",
            "point_cloud",
        )
        for candidate in candidates:
            if candidate in metadata.columns:
                return candidate
        return None

    @staticmethod
    def _infer_label_column(metadata: pd.DataFrame) -> str | None:
        candidates = ("label_idx", "label", "class", "category", "label_name")
        for candidate in candidates:
            if candidate in metadata.columns:
                return candidate
        return None

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for notebooks and sanity checks."""
        return {
            "metadata_csv": str(self.metadata_csv),
            "num_rows": len(self.metadata),
            "columns": list(self.metadata.columns),
            "label_column": self.label_column,
            "path_column": self.path_column,
            "num_classes": (
                int(self.metadata[self.label_column].nunique())
                if self.label_column is not None and not self.metadata.empty
                else 0
            ),
        }
