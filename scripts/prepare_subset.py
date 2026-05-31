#!/usr/bin/env python
"""Prepare a balanced subset scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/data.yaml")
    print("Subset preparation scaffold.")
    print(f"Target classes: {config['subset_classes']}")
    print(f"Samples per class: {config['samples_per_class']}")
    print("TODO: inspect labeled_dataset.csv, select balanced rows, and write data/splits files.")


if __name__ == "__main__":
    main()
