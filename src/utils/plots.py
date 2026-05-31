"""Plotting placeholders for report figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def save_metrics_csv(metrics: Sequence[Mapping[str, object]], path: str | Path) -> None:
    """Save simple metric rows as CSV without adding a plotting dependency yet."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(path, index=False)

