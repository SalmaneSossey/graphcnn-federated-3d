"""Deterministic split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFractions:
    """Train/validation/test split fractions."""

    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {total}")


def deterministic_split(
    dataframe: pd.DataFrame,
    label_column: str,
    fractions: SplitFractions = SplitFractions(),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create deterministic stratified-ish splits grouped by label."""
    fractions.validate()
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in dataframe.groupby(label_column):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(round(n_total * fractions.train))
        n_val = int(round(n_total * fractions.val))
        train_parts.append(dataframe.loc[indices[:n_train]])
        val_parts.append(dataframe.loc[indices[n_train : n_train + n_val]])
        test_parts.append(dataframe.loc[indices[n_train + n_val :]])

    return {
        "train": pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        "val": pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
        "test": pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True),
    }

