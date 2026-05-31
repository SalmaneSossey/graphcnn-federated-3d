"""Reusable training and evaluation loops."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    """Train for one epoch and return loss/accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for points, targets in dataloader:
        points = points.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(points)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> dict[str, float]:
    """Evaluate a model and return loss/accuracy."""
    model.eval()
    total_loss = 0.0
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    for points, targets in dataloader:
        points = points.to(device)
        targets = targets.to(device)
        logits = model(points)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    if not all_targets:
        return {"loss": 0.0, "accuracy": 0.0}
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    return {"loss": total_loss / targets.numel(), "accuracy": accuracy(logits, targets)}

