#!/usr/bin/env python
"""Prepare a balanced real point-cloud subset from Cap3D ShapeNet ZIP files.

The script never extracts a whole archive blindly. It selects IDs from
`labeled_dataset.csv`, searches ZIP files in `data/raw`, extracts only matching
`.ply` members, and writes deterministic split CSVs.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splits import SplitFractions, deterministic_split
from src.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    config = load_config("configs/data.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, default=Path(config["paths"]["metadata_csv"]))
    parser.add_argument("--raw-dir", type=Path, default=Path(config["paths"]["raw_dir"]))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/pointclouds"))
    parser.add_argument("--splits-dir", type=Path, default=Path(config["paths"]["splits_dir"]))
    parser.add_argument("--subset-classes", type=int, default=int(config["subset_classes"]))
    parser.add_argument("--samples-per-class", type=int, default=int(config["samples_per_class"]))
    parser.add_argument("--seed", type=int, default=int(config["seed"]))
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--skip-extract", action="store_true")
    return parser


def select_balanced_subset(
    metadata: pd.DataFrame,
    id_column: str,
    label_column: str,
    subset_classes: int,
    samples_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Select a balanced subset from the most frequent labels."""
    if id_column not in metadata.columns:
        raise ValueError(f"Missing ID column `{id_column}` in metadata.")
    if label_column not in metadata.columns:
        raise ValueError(f"Missing label column `{label_column}` in metadata.")

    frame = metadata[[id_column, label_column]].copy()
    frame[id_column] = frame[id_column].astype(str)
    frame["label_name"] = frame[label_column].astype(str)
    chosen_labels = frame["label_name"].value_counts().head(subset_classes).index.tolist()
    subset = (
        frame[frame["label_name"].isin(chosen_labels)]
        .sample(frac=1.0, random_state=seed)
        .groupby("label_name", group_keys=False)
        .head(samples_per_class)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    label_to_idx = {label: idx for idx, label in enumerate(sorted(subset["label_name"].unique()))}
    subset["label_idx"] = subset["label_name"].map(label_to_idx).astype(int)
    subset["file_path"] = subset[id_column] + ".ply"
    subset = subset.rename(columns={id_column: "id"})
    return subset[["id", "file_path", "label_name", "label_idx"]]


def index_zip_members(raw_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map point-cloud IDs to ZIP members for selective extraction."""
    index: dict[str, tuple[Path, str]] = {}
    zip_paths = sorted(raw_dir.rglob("*.zip"))
    print(f"Found {len(zip_paths)} ZIP file(s) under {raw_dir}.")
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or Path(member).suffix.lower() != ".ply":
                    continue
                point_id = Path(member).stem
                index.setdefault(point_id, (zip_path, member))
    return index


def extract_selected_pointclouds(subset: pd.DataFrame, raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Extract selected `.ply` files from available ZIP archives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_index = index_zip_members(raw_dir)
    if not zip_index:
        print(f"No ZIP files with .ply members found under {raw_dir}.")
        subset["extracted"] = False
        return subset

    extracted_flags: list[bool] = []
    for row in tqdm(subset.itertuples(index=False), total=len(subset), desc="Extracting PLY subset"):
        point_id = str(row.id)
        output_path = output_dir / f"{point_id}.ply"
        if output_path.exists():
            extracted_flags.append(True)
            continue
        match = zip_index.get(point_id)
        if match is None:
            extracted_flags.append(False)
            continue
        zip_path, member = match
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(member) as src, output_path.open("wb") as dst:
                dst.write(src.read())
        extracted_flags.append(True)

    subset = subset.copy()
    subset["extracted"] = extracted_flags
    return subset


def main() -> None:
    """Prepare selected subset and deterministic split CSVs."""
    args = build_parser().parse_args()
    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {args.metadata_csv}")

    metadata = pd.read_csv(args.metadata_csv)
    subset = select_balanced_subset(
        metadata=metadata,
        id_column=args.id_column,
        label_column=args.label_column,
        subset_classes=args.subset_classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    if not args.skip_extract:
        subset = extract_selected_pointclouds(subset, args.raw_dir, args.output_dir)
    else:
        subset["extracted"] = False

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.splits_dir / "selected_subset.csv", index=False)

    available = subset[subset["extracted"]].copy()
    if available.empty:
        print("No point clouds were extracted. Wrote selected_subset.csv only.")
        print("Place Cap3D ZIP files in data/raw and rerun without --skip-extract.")
        return

    splits = deterministic_split(
        available,
        label_column="label_idx",
        fractions=SplitFractions(train=0.70, val=0.15, test=0.15),
        seed=args.seed,
    )
    for split_name, split_df in splits.items():
        split_df.to_csv(args.splits_dir / f"{split_name}.csv", index=False)
        print(f"{split_name}: {len(split_df)} rows")
    print(f"Extracted {int(available['extracted'].sum())}/{len(subset)} selected point clouds.")
    print(f"Point-cloud root: {args.output_dir}")
    print(f"Split CSVs: {args.splits_dir}")


if __name__ == "__main__":
    main()
