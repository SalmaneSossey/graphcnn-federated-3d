#!/usr/bin/env python
"""Run horizontal federated learning with IID partitions scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/federated.yaml")
    print("HFL IID scaffold is ready.")
    print(f"Clients: {config['num_clients']}, rounds: {config['rounds']}")
    print("TODO: create client dataloaders and call manual FedAvg round loop.")


if __name__ == "__main__":
    main()
