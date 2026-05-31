"""Dataset scaffold tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset import ShapeNetPointCloudDataset


def test_dataset_reads_metadata_summary(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "file_path": ["class_a/example.ply"],
            "label": [0],
        }
    ).to_csv(metadata_path, index=False)

    dataset = ShapeNetPointCloudDataset(metadata_csv=metadata_path, pointcloud_root=tmp_path)
    summary = dataset.summary()
    assert len(dataset) == 1
    assert summary["path_column"] == "file_path"
    assert summary["label_column"] == "label"


def test_empty_dataset_when_metadata_missing(tmp_path: Path) -> None:
    dataset = ShapeNetPointCloudDataset(metadata_csv=tmp_path / "missing.csv")
    assert len(dataset) == 0

