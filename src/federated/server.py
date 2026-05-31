"""Manual horizontal federated learning server scaffold."""

from __future__ import annotations

import torch
from torch import nn

from src.federated.fedavg import aggregate_state_dicts


def apply_fedavg_round(
    global_model: nn.Module,
    client_states: list[dict[str, torch.Tensor]],
    client_sizes: list[int],
) -> nn.Module:
    """Aggregate client updates and load them into the global model."""
    aggregated = aggregate_state_dicts(client_states, client_sizes)
    global_model.load_state_dict(aggregated)
    return global_model

