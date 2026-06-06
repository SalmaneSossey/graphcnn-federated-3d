#!/usr/bin/env python
"""Parse and showcase differences in `data/splits/local_comparison_results.csv`.

Prints a ranked comparison table (with deltas against a chosen baseline) and
optionally saves a grouped bar chart of accuracy and MPCA per configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("data/splits/local_comparison_results.csv"),
        help="Path to the comparison results CSV.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Row label `configuration | model` to use as baseline. "
        "Defaults to the row with the highest test_accuracy.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("src/images/local_comparison.png"),
        help="Where to save the comparison bar chart (PNG).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the comparison chart.",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"configuration", "model", "test_loss", "test_accuracy", "test_mpca", "parameters"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df["label"] = df["configuration"].astype(str) + " | " + df["model"].astype(str)
    return df


def build_comparison(df: pd.DataFrame, baseline_label: str | None) -> tuple[pd.DataFrame, str]:
    ranked = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    if baseline_label is None:
        baseline_label = str(ranked.iloc[0]["label"])
    if baseline_label not in ranked["label"].values:
        raise ValueError(
            f"Baseline `{baseline_label}` not found. Available: {ranked['label'].tolist()}"
        )
    base = ranked.loc[ranked["label"] == baseline_label].iloc[0]
    comparison = ranked.copy()
    comparison["delta_accuracy"] = comparison["test_accuracy"] - float(base["test_accuracy"])
    comparison["delta_mpca"] = comparison["test_mpca"] - float(base["test_mpca"])
    comparison["delta_loss"] = comparison["test_loss"] - float(base["test_loss"])
    comparison["param_ratio"] = comparison["parameters"] / float(base["parameters"])
    return comparison, baseline_label


def print_table(comparison: pd.DataFrame, baseline_label: str) -> None:
    print(f"Baseline: {baseline_label}")
    print("=" * 110)
    view = comparison[
        [
            "configuration",
            "model",
            "test_accuracy",
            "delta_accuracy",
            "test_mpca",
            "delta_mpca",
            "test_loss",
            "delta_loss",
            "parameters",
            "param_ratio",
        ]
    ].copy()
    view["test_accuracy"] = view["test_accuracy"].map(lambda v: f"{v:.4f}")
    view["test_mpca"] = view["test_mpca"].map(lambda v: f"{v:.4f}")
    view["test_loss"] = view["test_loss"].map(lambda v: f"{v:.4f}")
    view["delta_accuracy"] = view["delta_accuracy"].map(lambda v: f"{v:+.4f}")
    view["delta_mpca"] = view["delta_mpca"].map(lambda v: f"{v:+.4f}")
    view["delta_loss"] = view["delta_loss"].map(lambda v: f"{v:+.4f}")
    view["param_ratio"] = view["param_ratio"].map(lambda v: f"x{v:.2f}")
    print(view.to_string(index=False))
    print("=" * 110)

    best = comparison.iloc[0]
    worst = comparison.iloc[-1]
    spread = float(best["test_accuracy"]) - float(worst["test_accuracy"])
    print(f"Best  : {best['label']}  (acc={best['test_accuracy']:.4f})")
    print(f"Worst : {worst['label']}  (acc={worst['test_accuracy']:.4f})")
    print(f"Spread: {spread:.4f} accuracy points across {len(comparison)} configurations.")


def plot_comparison(comparison: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = comparison["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels) + 2.0), 5.0))
    ax.bar(x - width / 2, comparison["test_accuracy"], width=width, label="Test Accuracy")
    ax.bar(x + width / 2, comparison["test_mpca"], width=width, label="Test MPCA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Configuration comparison (sorted by accuracy)")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    df = load_results(args.results_csv)
    comparison, baseline_label = build_comparison(df, args.baseline)
    print_table(comparison, baseline_label)
    if not args.no_plot:
        image_path = plot_comparison(comparison, args.output_image)
        print(f"Saved chart: {image_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
