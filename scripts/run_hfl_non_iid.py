#!/usr/bin/env python
"""Run horizontal federated learning with non-IID partitions scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/federated.yaml")
    print("HFL non-IID scaffold is ready.")
    print(f"Classes per client: {config['partition']['non_iid_classes_per_client']}")
    print("TODO: build label-skewed client dataloaders and call manual FedAvg.")


if __name__ == "__main__":
    main()
