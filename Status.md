# Project Status

Date: 2026-06-05

## Current State

The local repository has been pulled from GitHub and is on `main`.

Latest pulled commit:

```text
debaeea Created using Colab
```

The latest Colab notebook output confirms that the real Cap3D `.ply` workflow started successfully after `labeled_dataset.csv` was uploaded to Google Drive.

## Real Data Preparation

The notebook copied metadata from Drive:

```text
/content/drive/MyDrive/graphcnn-federated-3d/labeled_dataset.csv
```

It downloaded the Cap3D ZIP into temporary Colab runtime storage:

```text
PointCloud_zips_ShapeNet/compressed_pcs_00.zip
27.3G downloaded
```

It then extracted the selected real point-cloud subset:

```text
Extracted 2000/2000 selected point clouds.
train: 1400 rows
val: 300 rows
test: 300 rows
Prepared pointcloud count: 2000
Using real extracted PLY datasets.
```

Selected classes:

```text
table, chair, airplane, car, sofa, rifle, lamp, watercraft, bench, loudspeaker
```

The extracted subset and split CSVs were backed up to Google Drive:

```text
MyDrive/graphcnn-federated-3d/processed/pointclouds
MyDrive/graphcnn-federated-3d/splits
```

## Completed Real-Data Training

Centralized training completed for both models.

PointGCN:

```text
Final validation accuracy: 0.8433
Test loss: 0.360471
Test accuracy: 0.8900
Test MPCA: 0.8900
```

RS-CNN:

```text
Final validation accuracy: 0.8633
Test loss: 0.436849
Test accuracy: 0.876667
Test MPCA: 0.876667
```

The centralized stage was backed up to Drive:

```text
MyDrive/graphcnn-federated-3d/runs/after_centralized
```

## Where The Notebook Stopped

The saved notebook output stops during horizontal federated learning:

```text
C2 HFL IID round=01 val_acc=0.1000
C2 HFL IID round=02 val_acc=0.1333
```

There is no saved output for:

- HFL IID round 3
- HFL non-IID
- VFL
- distillation
- final comparison table
- final backup directory

No explicit Python error is saved in the notebook. The run likely stopped because the Colab session was interrupted, timed out, or was manually stopped.

## Notebook Resume Fix

The notebook was updated locally so future Colab runs can reuse the already extracted Drive subset instead of downloading the 27 GB ZIP again. It also restores the `after_centralized` outputs/checkpoints from Drive, so a reset runtime can skip centralized retraining and move on to HFL.

Changed setting:

```python
FORCE_REEXTRACT_SUBSET = False
```

The ZIP download cell now skips the download when it finds:

```text
2000 restored .ply files
train/val/test split CSVs
```

The restore cell now also loads:

```text
MyDrive/graphcnn-federated-3d/runs/after_centralized/outputs
MyDrive/graphcnn-federated-3d/runs/after_centralized/checkpoints
MyDrive/graphcnn-federated-3d/runs/after_centralized/reports/figures
```

The centralized cell now loads `pointgcn_centralized.pt` and `rscnn_centralized.pt` when available instead of training those models again.

The HFL non-IID loader was also fixed to partition the real loaded dataset instead of falling back to the synthetic dataset helper.

HFL now saves progress after each round to:

```text
MyDrive/graphcnn-federated-3d/runs/hfl_progress
```

If Colab resets during HFL again, the notebook can restore `hfl_progress` and resume the current HFL experiment from the next round.

## Local Git State

Current visible local changes:

```text
M notebooks/01_colab_safe_final_run.ipynb
?? Status.md
?? splits/
```

The `runs/` directory is ignored by Git and still contains the older local result bundle unless a newer Drive result archive is downloaded and extracted locally.

## Next Steps

1. Push or upload the updated notebook so Colab uses the resume fix.
2. In Colab, rerun the notebook from the beginning if the runtime reset. It should skip ZIP download, skip extraction, and load the centralized checkpoints from Drive.
3. Confirm it prints:

```text
Restored subset is ready: 2000 .ply files and split CSVs.
Skipping 26 GB ZIP download.
Using real extracted PLY datasets.
Loading restored checkpoint: checkpoints/pointgcn_centralized.pt
Loading restored checkpoint: checkpoints/rscnn_centralized.pt
```

4. Then continue from HFL onward.
5. Wait for VFL, distillation, comparison metrics, and final backup to complete.
6. Download the new final run from Drive and extract it locally.
7. Update the final report using the real-data metrics.
