#!/usr/bin/env python
"""Run vertical federated learning scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.federated.vfl import VFLEntityEncoder, VFLServerClassifier
from src.models.common import count_parameters


def main() -> None:
    entity_a = VFLEntityEncoder(input_channels=3)
    entity_b = VFLEntityEncoder(input_channels=3)
    server = VFLServerClassifier(num_classes=10)
    total = count_parameters(entity_a) + count_parameters(entity_b) + count_parameters(server)
    print(f"VFL scaffold parameters: {total:,}")
    print("Entity A owns XYZ, Entity B owns RGB, server sees only embeddings.")


if __name__ == "__main__":
    main()
