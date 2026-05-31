#!/usr/bin/env bash
set -euo pipefail

METADATA_PATH="data/metadata/labeled_dataset.csv"

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC is not installed. Install it with: python -m pip install dvc"
  exit 1
fi

if [ ! -d ".dvc" ]; then
  echo "DVC is not initialized. Run: bash scripts/setup_dvc.sh"
  exit 1
fi

if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing $METADATA_PATH"
  echo "Move your local labeled_dataset.csv there, then rerun this script."
  exit 1
fi

dvc add "$METADATA_PATH"
git add "${METADATA_PATH}.dvc" data/metadata/.gitignore .gitignore

echo
echo "Metadata is now tracked by DVC."
echo "Next commands:"
echo "  git commit -m 'Track labeled metadata with DVC'"
echo "  dvc push"

