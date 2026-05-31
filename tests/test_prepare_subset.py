"""Tests for real subset preparation helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.prepare_subset import index_zip_members


def test_index_zip_members_searches_nested_zip_files(tmp_path: Path) -> None:
    nested_dir = tmp_path / "PointCloud_zips_ShapeNet"
    nested_dir.mkdir()
    zip_path = nested_dir / "compressed_pcs_00.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("some/folder/example_id.ply", "ply\n")

    index = index_zip_members(tmp_path)

    assert "example_id" in index
    assert index["example_id"] == (zip_path, "some/folder/example_id.ply")

