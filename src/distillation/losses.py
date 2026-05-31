"""Knowledge distillation losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Combine CE loss with KL distillation loss.

    Teacher logits are detached inside the function so callers can pass logits
    computed under `torch.no_grad()` or regular tensors safely.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    teacher_logits = teacher_logits.detach()
    ce_loss = F.cross_entropy(student_logits, targets)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return (1.0 - alpha) * ce_loss + alpha * (temperature**2) * kl_loss

