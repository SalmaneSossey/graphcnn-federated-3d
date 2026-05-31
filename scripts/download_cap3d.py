#!/usr/bin/env python
"""Download selected Cap3D ShapeNet files without importing training code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.download_cap3d import download_cap3d_shapenet


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected Cap3D ShapeNet files.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    paths = download_cap3d_shapenet(args.data_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
