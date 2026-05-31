#!/usr/bin/env bash
set -euo pipefail

if ! command -v dvc >/dev/null 2>&1; then
  echo "DVC is not installed. Install it with: python -m pip install dvc"
  exit 1
fi

if [ ! -d ".dvc" ]; then
  dvc init
else
  echo "DVC is already initialized."
fi

mkdir -p data/metadata data/raw data/processed data/splits data/cache checkpoints outputs runs
touch data/metadata/.gitkeep data/raw/.gitkeep data/processed/.gitkeep data/splits/.gitkeep data/cache/.gitkeep

echo
echo "DVC is ready locally. Configure a remote manually before running dvc push."
echo
echo "Examples:"
echo "  # Local remote"
echo "  dvc remote add -d localstore /absolute/path/to/dvc-store"
echo
echo "  # Google Drive remote"
echo "  dvc remote add -d gdrive gdrive://<folder-id>"
echo
echo "  # S3 remote"
echo "  dvc remote add -d s3remote s3://<bucket>/<prefix>"
echo
echo "Then run:"
echo "  git add .dvc .dvcignore .gitignore"
echo "  git commit -m 'Initialize DVC'"
echo "  dvc push"

