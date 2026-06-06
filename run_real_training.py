#!/usr/bin/env python
"""Execute the real 3D Point Cloud training pipeline locally."""

import copy
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

# Import core modules from the repository package
from src.data.dataset import ShapeNetPointCloudDataset
from src.distillation.losses import distillation_loss
from src.federated.fedavg import aggregate_state_dicts
from src.federated.vfl import VFLEntityEncoder, VFLServerClassifier, split_xyz_rgb
from src.models.common import count_parameters
from src.models.pointgcn import PointGCN
from src.models.rscnn import RSCNN
from src.models.student import StudentPointCloudMLP
from src.training.metrics import accuracy, mean_per_class_accuracy

# --- Configuration ---
SEED = 42
NUM_POINTS = 1024
BATCH_SIZE = 32
POINTCLOUD_ROOT = "data/processed/pointclouds"

# Reduced defaults for local CPU runs (Increase for final papers/reports)
CENTRALIZED_EPOCHS = 5  # Set to 25 for full training
HFL_ROUNDS = 3  # Set to 10 for full training
HFL_LOCAL_EPOCHS = 1  # Set to 2 for full training
VFL_EPOCHS = 5  # Set to 20 for full training
DISTILL_EPOCHS = 5  # Set to 20 for full training
NUM_CLIENTS = 4

LOG_DIR = Path("logs")
CHECKPOINT_DIR = Path("checkpoints")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = LOG_DIR / f"run_real_training_{RUN_TIMESTAMP}.log"


# --- Logging setup ---
def _configure_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_real_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


log = _configure_logger()


# --- Graceful interruption support ---
class TrainingInterrupted(Exception):
    """Raised internally when SIGINT is received and training must stop."""


STOP_EVENT = threading.Event()
LIVE_MODELS: dict[str, nn.Module] = {}


def _handle_sigint(signum, frame) -> None:  # noqa: ARG001
    if STOP_EVENT.is_set():
        log.warning("Second interrupt received; exiting immediately.")
        sys.exit(130)
    STOP_EVENT.set()
    log.warning("Interrupt received. Finishing current batch then saving weights...")


def _check_stop() -> None:
    if STOP_EVENT.is_set():
        raise TrainingInterrupted()


def register_model(name: str, model: nn.Module) -> nn.Module:
    LIVE_MODELS[name] = model
    return model


def save_all_live_models(suffix: str = "interrupted") -> list[Path]:
    if not LIVE_MODELS:
        return []
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for name, model in LIVE_MODELS.items():
        path = CHECKPOINT_DIR / f"{name}_{suffix}_{RUN_TIMESTAMP}.pt"
        try:
            torch.save(model.state_dict(), path)
            saved.append(path)
            log.info("Saved checkpoint: %s", path)
        except Exception as exc:  # pragma: no cover - defensive
            log.error("Failed to save %s: %s", name, exc)
    return saved


# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", device)
log.info("Logging to: %s", LOG_PATH)

# --- Helper Functions ---
criterion = nn.CrossEntropyLoss()
experiment_rows: list[dict] = []


def evaluate(model, loader, num_classes):
    model.eval()
    total_loss = 0.0
    logits_all, targets_all = [], []
    with torch.no_grad():
        for points, targets in loader:
            points = points.to(device)
            targets = targets.to(device)
            logits = model(points)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            logits_all.append(logits.cpu())
            targets_all.append(targets.cpu())
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    return {
        "loss": total_loss / max(len(targets), 1),
        "accuracy": accuracy(logits, targets),
        "mpca": mean_per_class_accuracy(logits, targets, num_classes=num_classes),
    }


def train_model(
    model, train_loader, val_loader, epochs, num_classes, lr=1e-3, phase: str = "train"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log.info("[%s] Starting training: epochs=%d lr=%g", phase, epochs, lr)
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"{phase} epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_examples = 0
        batch_bar = tqdm(
            train_loader, desc=f"{phase} e{epoch:02d}", unit="batch", leave=False
        )
        for points, targets in batch_bar:
            _check_stop()
            points = points.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_examples += targets.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        batch_bar.close()
        val_metrics = evaluate(model, val_loader, num_classes)
        elapsed = time.perf_counter() - epoch_start
        train_loss = total_loss / max(total_examples, 1)
        log.info(
            "[%s] Epoch %02d/%02d | Train Loss: %.4f | Val Acc: %.4f | Val MPCA: %.4f | Time: %.2fs",
            phase,
            epoch,
            epochs,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["mpca"],
            elapsed,
        )
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            t=f"{elapsed:.1f}s",
        )
    epoch_bar.close()
    return model


def record_result(config_name, model_name, metrics, params):
    row = {
        "configuration": config_name,
        "model": model_name,
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_mpca": metrics["mpca"],
        "parameters": params,
    }
    experiment_rows.append(row)


class VFLSystem(nn.Module):
    def __init__(self, embedding_dim=64, num_classes=10):
        super().__init__()
        self.entity_a = VFLEntityEncoder(input_channels=3, embedding_dim=embedding_dim)
        self.entity_b = VFLEntityEncoder(input_channels=3, embedding_dim=embedding_dim)
        self.server = VFLServerClassifier(
            embedding_dim=embedding_dim, num_classes=num_classes
        )

    def forward(self, points):
        xyz, rgb = split_xyz_rgb(points)
        return self.server(self.entity_a(xyz), self.entity_b(rgb))


def main() -> None:
    pipeline_start = time.perf_counter()

    # --- 1. Load Real Extracted Splits ---
    split_paths = {
        name: Path(f"data/splits/{name}.csv") for name in ["train", "val", "test"]
    }
    train_df = pd.read_csv(split_paths["train"])
    num_classes = int(
        pd.concat([pd.read_csv(p) for p in split_paths.values()])["label_idx"].nunique()
    )
    log.info("Detected num_classes=%d", num_classes)

    train_dataset = ShapeNetPointCloudDataset(
        split_paths["train"], POINTCLOUD_ROOT, NUM_POINTS, label_column="label_idx"
    )
    val_dataset = ShapeNetPointCloudDataset(
        split_paths["val"], POINTCLOUD_ROOT, NUM_POINTS, label_column="label_idx"
    )
    test_dataset = ShapeNetPointCloudDataset(
        split_paths["test"], POINTCLOUD_ROOT, NUM_POINTS, label_column="label_idx"
    )
    log.info(
        "Dataset sizes | train=%d val=%d test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. C1: Centralized Training ---
    log.info(">>> Phase 1: Training Centralized PointGCN <<<")
    pointgcn = register_model(
        "pointgcn",
        PointGCN(num_classes=num_classes, input_channels=6, hidden_dim=64),
    )
    pointgcn = train_model(
        pointgcn,
        train_loader,
        val_loader,
        CENTRALIZED_EPOCHS,
        num_classes,
        phase="C1-PointGCN",
    )
    metrics_pointgcn = evaluate(pointgcn, test_loader, num_classes)
    record_result(
        "C1 Centralized", "PointGCN", metrics_pointgcn, count_parameters(pointgcn)
    )

    log.info(">>> Phase 1: Training Centralized RS-CNN <<<")
    rscnn = register_model(
        "rscnn",
        RSCNN(num_classes=num_classes, input_channels=6, hidden_dim=64),
    )
    rscnn = train_model(
        rscnn,
        train_loader,
        val_loader,
        CENTRALIZED_EPOCHS,
        num_classes,
        phase="C1-RSCNN",
    )
    metrics_rscnn = evaluate(rscnn, test_loader, num_classes)
    record_result("C1 Centralized", "RS-CNN", metrics_rscnn, count_parameters(rscnn))

    # --- 3. C2: Horizontal FL (IID) ---
    log.info(">>> Phase 2: Horizontal FL (IID Partitions) <<<")
    global_model = register_model(
        "hfl_iid_global",
        PointGCN(num_classes=num_classes, input_channels=6, hidden_dim=64).to(device),
    )

    # Partition indices uniformly (IID)
    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    client_shards = np.array_split(indices, NUM_CLIENTS)
    iid_loaders = [
        DataLoader(
            Subset(train_dataset, shard.tolist()), batch_size=BATCH_SIZE, shuffle=True
        )
        for shard in client_shards
    ]

    rounds_bar = tqdm(range(1, HFL_ROUNDS + 1), desc="HFL-IID rounds", unit="round")
    for r in rounds_bar:
        round_start = time.perf_counter()
        client_states, client_sizes = [], []
        for c_idx, loader in enumerate(iid_loaders):
            local_model = copy.deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
            for _ in range(HFL_LOCAL_EPOCHS):
                local_model.train()
                for pts, targets in tqdm(
                    loader, desc=f"r{r:02d} c{c_idx}", unit="batch", leave=False
                ):
                    _check_stop()
                    pts, targets = pts.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(local_model(pts), targets)
                    loss.backward()
                    optimizer.step()
            client_states.append(
                {k: v.cpu() for k, v in local_model.state_dict().items()}
            )
            client_sizes.append(len(loader.dataset))

        aggregated = aggregate_state_dicts(client_states, client_sizes)
        global_model.load_state_dict(aggregated)
        val_acc = evaluate(global_model, val_loader, num_classes)["accuracy"]
        elapsed = time.perf_counter() - round_start
        log.info(
            "[HFL-IID] Round %02d/%02d | Val Acc: %.4f | Time: %.2fs",
            r,
            HFL_ROUNDS,
            val_acc,
            elapsed,
        )
        rounds_bar.set_postfix(val_acc=f"{val_acc:.4f}", t=f"{elapsed:.1f}s")
    rounds_bar.close()

    metrics_hfl_iid = evaluate(global_model, test_loader, num_classes)
    record_result(
        "C2 HFL IID",
        "PointGCN (FedAvg)",
        metrics_hfl_iid,
        count_parameters(global_model),
    )

    # --- 4. C3: Horizontal FL (Non-IID) ---
    log.info(">>> Phase 3: Horizontal FL (Non-IID Partitions) <<<")
    global_non_iid = register_model(
        "hfl_non_iid_global",
        PointGCN(num_classes=num_classes, input_channels=6, hidden_dim=64).to(device),
    )

    # Create label-skewed client sets using localized disk structures
    sorted_frame = train_df.sort_values(["label_idx"]).reset_index(drop=True)
    shards = np.array_split(np.arange(len(sorted_frame)), NUM_CLIENTS)
    non_iid_loaders = []
    Path("data/splits/temp_clients").mkdir(parents=True, exist_ok=True)

    for c_idx, shard in enumerate(shards):
        client_frame = sorted_frame.iloc[shard].reset_index(drop=True)
        temp_csv = f"data/splits/temp_clients/client_{c_idx}.csv"
        client_frame.to_csv(temp_csv, index=False)
        client_dataset = ShapeNetPointCloudDataset(
            temp_csv, POINTCLOUD_ROOT, NUM_POINTS, label_column="label_idx"
        )
        non_iid_loaders.append(
            DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
        )

    rounds_bar = tqdm(range(1, HFL_ROUNDS + 1), desc="HFL-NonIID rounds", unit="round")
    for r in rounds_bar:
        round_start = time.perf_counter()
        client_states, client_sizes = [], []
        for c_idx, loader in enumerate(non_iid_loaders):
            local_model = copy.deepcopy(global_non_iid).to(device)
            optimizer = torch.optim.Adam(local_model.parameters(), lr=1e-3)
            for _ in range(HFL_LOCAL_EPOCHS):
                local_model.train()
                for pts, targets in tqdm(
                    loader, desc=f"r{r:02d} c{c_idx}", unit="batch", leave=False
                ):
                    _check_stop()
                    pts, targets = pts.to(device), targets.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(local_model(pts), targets)
                    loss.backward()
                    optimizer.step()
            client_states.append(
                {k: v.cpu() for k, v in local_model.state_dict().items()}
            )
            client_sizes.append(len(loader.dataset))

        aggregated = aggregate_state_dicts(client_states, client_sizes)
        global_non_iid.load_state_dict(aggregated)
        val_acc = evaluate(global_non_iid, val_loader, num_classes)["accuracy"]
        elapsed = time.perf_counter() - round_start
        log.info(
            "[HFL-NonIID] Round %02d/%02d | Val Acc: %.4f | Time: %.2fs",
            r,
            HFL_ROUNDS,
            val_acc,
            elapsed,
        )
        rounds_bar.set_postfix(val_acc=f"{val_acc:.4f}", t=f"{elapsed:.1f}s")
    rounds_bar.close()

    metrics_hfl_non_iid = evaluate(global_non_iid, test_loader, num_classes)
    record_result(
        "C3 HFL Non-IID",
        "PointGCN (FedAvg)",
        metrics_hfl_non_iid,
        count_parameters(global_non_iid),
    )

    # --- 5. Vertical FL (XYZ/RGB Feature Split) ---
    log.info(">>> Phase 4: Vertical FL Simulation <<<")
    vfl_model = register_model(
        "vfl_model",
        VFLSystem(embedding_dim=64, num_classes=num_classes).to(device),
    )
    vfl_model = train_model(
        vfl_model, train_loader, val_loader, VFL_EPOCHS, num_classes, phase="VFL"
    )
    metrics_vfl = evaluate(vfl_model, test_loader, num_classes)
    record_result(
        "VFL Feature Split",
        "XYZ/RGB Embedding Concatenation",
        metrics_vfl,
        count_parameters(vfl_model),
    )

    # --- 6. C4: Knowledge Distillation ---
    log.info(">>> Phase 5: Knowledge Distillation <<<")
    teacher = pointgcn.eval()
    student = register_model(
        "student",
        StudentPointCloudMLP(
            num_classes=num_classes, input_channels=6, hidden_dim=16
        ).to(device),
    )
    optimizer_student = torch.optim.Adam(student.parameters(), lr=1e-3)

    epoch_bar = tqdm(range(1, DISTILL_EPOCHS + 1), desc="Distill epochs", unit="epoch")
    for epoch in epoch_bar:
        epoch_start = time.perf_counter()
        student.train()
        total_loss = 0.0
        batch_bar = tqdm(
            train_loader, desc=f"distill e{epoch:02d}", unit="batch", leave=False
        )
        for pts, targets in batch_bar:
            _check_stop()
            pts, targets = pts.to(device), targets.to(device)
            optimizer_student.zero_grad(set_to_none=True)
            student_logits = student(pts)
            with torch.no_grad():
                teacher_logits = teacher(pts)
            loss = distillation_loss(
                student_logits, teacher_logits, targets, alpha=0.5, temperature=4.0
            )
            loss.backward()
            optimizer_student.step()
            total_loss += loss.item() * targets.size(0)
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        batch_bar.close()
        val_acc = evaluate(student, val_loader, num_classes)["accuracy"]
        elapsed = time.perf_counter() - epoch_start
        distill_loss = total_loss / max(len(train_dataset), 1)
        log.info(
            "[Distill] Epoch %02d/%02d | Loss: %.4f | Val Acc: %.4f | Time: %.2fs",
            epoch,
            DISTILL_EPOCHS,
            distill_loss,
            val_acc,
            elapsed,
        )
        epoch_bar.set_postfix(
            loss=f"{distill_loss:.4f}", val_acc=f"{val_acc:.4f}", t=f"{elapsed:.1f}s"
        )
    epoch_bar.close()

    metrics_student = evaluate(student, test_loader, num_classes)
    record_result(
        "C4 Distillation",
        "Student (Alpha=0.5, Temp=4.0)",
        metrics_student,
        count_parameters(student),
    )

    # --- 7. Save and Report ---
    total_elapsed = time.perf_counter() - pipeline_start
    log.info("=================== FINAL PERFORMANCE TABLE ===================")
    results_df = pd.DataFrame(experiment_rows)
    results_csv = Path("data/splits/local_comparison_results.csv")
    results_df.to_csv(results_csv, index=False)
    for line in results_df.to_string(index=False).splitlines():
        log.info(line)
    log.info("===============================================================")
    log.info("Results CSV: %s", results_csv)
    log.info("Total pipeline time: %.2fs", total_elapsed)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    exit_code = 0
    try:
        main()
    except TrainingInterrupted:
        log.warning("Training interrupted by user. Saving live model weights...")
        save_all_live_models(suffix="interrupted")
        log.warning("Exit after interruption. Log saved to %s", LOG_PATH)
        exit_code = 130
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt outside training loop. Saving weights...")
        save_all_live_models(suffix="interrupted")
        exit_code = 130
    except Exception:
        log.exception("Unhandled exception during pipeline.")
        save_all_live_models(suffix="crash")
        exit_code = 1
    finally:
        logging.shutdown()
    sys.exit(exit_code)
