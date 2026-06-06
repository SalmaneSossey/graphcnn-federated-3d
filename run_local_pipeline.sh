#!/usr/bin/env bash

set -e

echo "===================================================="
echo "      Starting 3D Point Cloud Local Pipeline        "
echo "===================================================="


if [ -d ".venv" ]; then
    echo "Found local virtual environment. Activating..."
    source .venv/bin/activate
else
    echo "Warning: '.venv' folder not found. Proceeding with default system Python."
fi


METADATA_CSV="data/metadata/labeled_dataset.csv"
if [ ! -f "$METADATA_CSV" ]; then
    echo "Error: Private metadata file missing at: $METADATA_CSV"
    echo "Please copy your labeled_dataset.csv to data/metadata/ before running."
    echo "Aborting run."
    exit 1
fi
echo "✔ Metadata file found at $METADATA_CSV"



ZIP_COUNT=$(find data/raw -maxdepth 3 -name "*.zip" 2>/dev/null | wc -l || echo 0)
PLY_COUNT=$(find data/processed/pointclouds -name "*.ply" 2>/dev/null | wc -l || echo 0)

if [ "$PLY_COUNT" -gt 0 ]; then
    echo "✔ Found $PLY_COUNT extracted .ply files under data/processed/pointclouds."
    echo "  Extraction step will be skipped."
elif [ "$ZIP_COUNT" -gt 0 ]; then
    echo "✔ Found $ZIP_COUNT raw ZIP archive(s) under data/raw/."
    echo "  Subset extraction is ready to proceed."
else
    echo "Error: No raw or pre-processed data found!"
    echo "  Expected: Either .zip files under 'data/raw/' OR extracted .ply files under 'data/processed/pointclouds/'."
    echo "  Note: Download is disabled. Please place your raw data files manually."
    echo "Aborting run."
    exit 1
fi


echo ""
echo ">>> Step 1: Preprocessing and Stratified Split <<<"
python scripts/prepare_subset.py --subset-classes 10 --samples-per-class 200


echo ""
echo ">>> Step 2: Centralized Baselines <<<"
echo "Running Centralized PointGCN..."
python scripts/train_pointgcn_centralized.py

echo "Running Centralized RS-CNN..."
python scripts/train_rscnn_centralized.py


echo ""
echo ">>> Step 3: Federated Learning Simulations <<<"
echo "Running HFL (IID)..."
python scripts/run_hfl_iid.py

echo "Running HFL (Non-IID)..."
python scripts/run_hfl_non_iid.py

echo "Running Vertical FL (VFL)..."
python scripts/run_vfl.py


echo ""
echo ">>> Step 4: Knowledge Distillation <<<"
python scripts/run_distillation.py

echo "===================================================="
echo "          Pipeline Execution Completed              "
echo "===================================================="
