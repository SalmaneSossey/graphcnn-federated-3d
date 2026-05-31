"""Metrics for classification experiments."""

from __future__ import annotations

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy from logits and integer labels."""
    if targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def mean_per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int | None = None,
) -> float:
    """Compute the mean accuracy over classes present in `targets`."""
    if targets.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    if num_classes is None:
        num_classes = int(targets.max().item()) + 1

    class_acc: list[torch.Tensor] = []
    for class_id in range(num_classes):
        mask = targets == class_id
        if mask.any():
            class_acc.append((preds[mask] == targets[mask]).float().mean())
    if not class_acc:
        return 0.0
    return torch.stack(class_acc).mean().item()

