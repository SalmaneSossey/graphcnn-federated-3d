"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and resolve `device: auto` when present."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    def resolve_device(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: resolve_device(value) for key, value in node.items()}
        if node == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return node

    return resolve_device(config)

