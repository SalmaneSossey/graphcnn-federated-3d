# 3D Point Cloud Classification with PointGCN, RS-CNN, Federated Learning, VFL, and Distillation

Clean university ML project scaffold for 3D object classification using point clouds from ShapeNet via Cap3D.

Each object is represented as a `.ply` point cloud. Points are expected to contain XYZ coordinates and optionally RGB color values, producing model inputs of shape `[B, N, C]`, where `N=1024` by default and `C=6` for XYZRGB.

## Project Overview

The repository is designed for lightweight, reproducible experiments on Google Colab T4. The first implementation includes placeholders for the selected architectures and real utilities for configuration, metrics, checkpoints, DVC workflows, FedAvg, and distillation loss.

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

This file is ignored by Git and should be tracked with DVC.

## DVC Workflow

Initialize DVC locally:

```bash
bash scripts/setup_dvc.sh
```

Add the label metadata to DVC:

```bash
bash scripts/add_labeled_dataset_to_dvc.sh
```

Configure a remote manually before pushing data:

```bash
dvc remote add -d localstore /path/to/dvc-store
# or: dvc remote add -d gdrive gdrive://<folder-id>
# or: dvc remote add -d s3remote s3://<bucket>/<prefix>
dvc push
```

Warning: do not commit raw datasets, extracted `.ply` files, processed datasets, checkpoints, `outputs/`, `runs/`, or `wandb/` to GitHub.

## Colab T4 Workflow

1. Open `notebooks/00_colab_setup.ipynb`.
2. Clone the repository and install `requirements-colab.txt`.
3. Configure DVC credentials or pull from a configured remote.
4. Download only the Cap3D files you need:

```bash
python scripts/download_cap3d.py --data-dir data/raw
```

Do not unzip the full dataset blindly on Colab. Build a balanced representative subset first.

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
- `data/`: DVC-managed data roots and tracked README/placeholders.
- `notebooks/`: Colab starter notebooks.
- `src/`: reusable Python package.
- `scripts/`: command-line workflows.
- `reports/`: report assets and French template.
- `tests/`: smoke tests for imports, shapes, FedAvg, dataset, and distillation.
