"""Download helpers for selected Cap3D ShapeNet files.

This module never downloads data on import. Call `download_cap3d_shapenet`
explicitly from a script or notebook.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


CAP3D_REPO_ID = "tiange/Cap3D"
CAP3D_FILES = {
    "pointcloud_zip": "PointCloud_zips_ShapeNet/compressed_pcs_00.zip",
    "files_info": "PointCloud_zips_ShapeNet/compressed_files_info.pkl",
    "captions_csv": "Cap3D_automated_ShapeNet.csv",
}


def download_cap3d_shapenet(data_dir: str | Path) -> dict[str, Path]:
    """Download selected Cap3D ShapeNet files into `data_dir`.

    The full dataset is large. This helper only fetches the requested files and
    leaves extraction/subsetting to later explicit preprocessing steps.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for key, filename in CAP3D_FILES.items():
        path = hf_hub_download(
            repo_id=CAP3D_REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=data_dir,
        )
        downloaded[key] = Path(path)
    return downloaded

