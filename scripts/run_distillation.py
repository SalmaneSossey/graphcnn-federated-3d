#!/usr/bin/env python
"""Run knowledge distillation scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation.train_student import run_distillation_training
from src.models.common import count_parameters
from src.models.student import StudentPointCloudMLP
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/distillation.yaml")
    student = StudentPointCloudMLP(
        num_classes=config["num_classes"],
        input_channels=config["input_channels"],
        hidden_dim=config["student_hidden_dim"],
    )
    print(f"Student scaffold parameters: {count_parameters(student):,}")
    run_distillation_training()


if __name__ == "__main__":
    main()
