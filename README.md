# 3D Point Cloud Classification with PointGCN, RS-CNN, Federated Learning, VFL, and Distillation

Clean university ML project scaffold for 3D object classification using point clouds from ShapeNet via Cap3D.

Each object is represented as a `.ply` point cloud. Points are expected to contain XYZ coordinates and optionally RGB color values, producing model inputs of shape `[B, N, C]`, where `N=1024` by default and `C=6` for XYZRGB.

## Project Overview

The repository is designed for lightweight, reproducible experiments on Google Colab T4. The first implementation includes placeholders for the selected architectures and real utilities for configuration, metrics, checkpoints, FedAvg, and distillation loss.

Chosen model families:

- PointGCN
- RS-CNN
- PointCloudMLP sanity-check baseline

Assignment configurations:

- C1: centralized baseline
- C2: horizontal federated learning with IID partitions
- C3: horizontal federated learning with non-IID partitions
- C4: knowledge distillation
- VFL: vertical federated learning with XYZ and RGB owned by separate entities

## Dataset

Official dataset sources:

- Hugging Face dataset repo: `tiange/Cap3D`
- Main dataset page: <https://huggingface.co/datasets/tiange/Cap3D>
- ShapeNet point-cloud folder: <https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_zips_ShapeNet>
- Main ZIP file: `PointCloud_zips_ShapeNet/compressed_pcs_00.zip`
- Info file: `PointCloud_zips_ShapeNet/compressed_files_info.pkl`
- Optional metadata/captions file: `Cap3D_automated_ShapeNet.csv`

Local label metadata should live at:

```bash
data/metadata/labeled_dataset.csv
```

This file is ignored by Git. Keep it in Google Drive or another private storage location, then copy it into the repo when running experiments.

## Data Workflow

Keep GitHub code-only. Store private metadata and heavy data in Google Drive:

```text
MyDrive/graphcnn-federated-3d/labeled_dataset.csv
MyDrive/graphcnn-federated-3d/raw/
MyDrive/graphcnn-federated-3d/processed/
```

For local runs, copy metadata into the ignored project path:

```bash
mkdir -p data/metadata
cp /path/to/labeled_dataset.csv data/metadata/labeled_dataset.csv
```

Warning: do not commit label metadata, raw datasets, extracted `.ply` files, processed datasets, checkpoints, `outputs/`, `runs/`, or `wandb/` to GitHub.

## Colab T4 Workflow

Recommended one-notebook workflow:

1. Open `notebooks/00_colab_run_all_google_drive.ipynb` in Google Colab.
2. Select a T4 GPU runtime.
3. Put `labeled_dataset.csv` in `MyDrive/graphcnn-federated-3d/labeled_dataset.csv`.
4. Use `Runtime > Run all`.

The notebook clones or updates the GitHub repo, mounts Google Drive, installs Colab-safe dependencies, copies the label metadata into the ignored local data folder, prepares a balanced subset, runs fast training experiments, and saves comparison results. It covers centralized PointGCN/RS-CNN, manual HFL IID/non-IID with FedAvg, VFL with XYZ/RGB split embeddings, and knowledge distillation.

Download selected Cap3D files only when you are ready:

```bash
python scripts/download_cap3d.py --data-dir data/raw
```

Colab already includes a CUDA-enabled PyTorch build, so avoid reinstalling
`torch` there unless you have a specific reason.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest
```

Run placeholder training entrypoints:

```bash
python scripts/train_pointgcn_centralized.py
python scripts/train_rscnn_centralized.py
python scripts/run_hfl_iid.py
python scripts/run_hfl_non_iid.py
python scripts/run_vfl.py
python scripts/run_distillation.py
```

## Repository Layout

- `configs/`: experiment defaults.
- `data/`: ignored local data roots and tracked README/placeholders.
- `notebooks/`: the single Colab run-all workflow.
- `src/`: reusable Python package.
- `scripts/`: command-line workflows.
- `reports/`: report assets and French template.
- `tests/`: smoke tests for imports, shapes, FedAvg, dataset, and distillation.
