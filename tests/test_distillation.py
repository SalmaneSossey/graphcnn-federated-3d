"""Knowledge distillation tests."""

from __future__ import annotations

import torch

from src.distillation.losses import distillation_loss


def test_distillation_loss_returns_scalar_tensor() -> None:
    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    targets = torch.tensor([0, 1, 2, 3])
    loss = distillation_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=2.0)
    assert loss.ndim == 0
    loss.backward()
    assert student_logits.grad is not None

