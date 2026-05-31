"""Dataset scaffold tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset import ShapeNetPointCloudDataset


def _write_ascii_ply(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "0 1 0 0 0 255",
            ]
        ),
        encoding="utf-8",
    )


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


def test_dataset_loads_ascii_ply_with_string_labels(tmp_path: Path) -> None:
    point_root = tmp_path / "pointclouds"
    point_root.mkdir()
    _write_ascii_ply(point_root / "example.ply")
    metadata_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "file_path": ["example.ply"],
            "label_name": ["chair"],
        }
    ).to_csv(metadata_path, index=False)

    dataset = ShapeNetPointCloudDataset(
        metadata_csv=metadata_path,
        pointcloud_root=point_root,
        num_points=8,
        label_column="label_name",
    )
    points, label = dataset[0]
    assert points.shape == (8, 6)
    assert label == 0
