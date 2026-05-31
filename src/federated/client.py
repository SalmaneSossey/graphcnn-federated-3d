"""Horizontal federated learning client helpers."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.loops import train_one_epoch


@dataclass
class ClientUpdate:
    """Payload returned by a client after local training."""

    state_dict: dict[str, torch.Tensor]
    num_examples: int


def train_client(
    global_model: nn.Module,
    dataloader: DataLoader,
    optimizer_cls: type[torch.optim.Optimizer],
    criterion: nn.Module,
    device: str | torch.device,
    local_epochs: int = 1,
    learning_rate: float = 1e-3,
) -> ClientUpdate:
    """Train a copied global model locally and return its state and data size."""
    local_model = copy.deepcopy(global_model).to(device)
    optimizer = optimizer_cls(local_model.parameters(), lr=learning_rate)
    for _ in range(local_epochs):
        train_one_epoch(local_model, dataloader, optimizer, criterion, device)
    return ClientUpdate(
        state_dict={key: value.detach().cpu() for key, value in local_model.state_dict().items()},
        num_examples=len(dataloader.dataset),
    )

