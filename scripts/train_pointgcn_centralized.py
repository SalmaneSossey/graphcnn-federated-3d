#!/usr/bin/env python
"""Run centralized PointGCN training scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pointgcn import PointGCN
from src.models.common import count_parameters
from src.training.train_centralized import run_centralized_training
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/pointgcn.yaml")
    model_cfg = config["model"]
    model = PointGCN(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        hidden_dim=model_cfg["hidden_dim"],
    )
    print(f"PointGCN scaffold parameters: {count_parameters(model):,}")
    run_centralized_training()


if __name__ == "__main__":
    main()
