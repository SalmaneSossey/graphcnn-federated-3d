# Data Directory

This project keeps large data outside Git. Use DVC to version local artifacts and configure a remote before sharing the project.

Expected layout:

- `metadata/labeled_dataset.csv`: label metadata tracked with DVC.
- `raw/`: downloaded Cap3D/ShapeNet ZIPs and extracted point clouds.
- `processed/`: sampled, normalized, or cached tensors.
- `splits/`: deterministic train/validation/test split files.
- `cache/`: temporary preprocessing cache.

Do not commit raw point clouds, extracted `.ply` files, processed datasets, checkpoints, or outputs directly to GitHub.

