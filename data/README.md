# Data Directory

This project keeps large data outside Git. Store private data in Google Drive or another external location, then copy it into the expected local folders when running experiments.

Expected layout:

- `metadata/labeled_dataset.csv`: private label metadata copied locally from Google Drive.
- `raw/`: downloaded Cap3D/ShapeNet ZIPs and extracted point clouds.
- `processed/`: sampled, normalized, or cached tensors.
- `splits/`: deterministic train/validation/test split files.
- `cache/`: temporary preprocessing cache.

Do not commit label metadata, raw point clouds, extracted `.ply` files, processed datasets, checkpoints, or outputs directly to GitHub.
