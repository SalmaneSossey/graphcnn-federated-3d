"""FedAvg tests."""

from __future__ import annotations

import torch

from src.federated.fedavg import aggregate_state_dicts


def test_fedavg_weighted_float_tensors() -> None:
    states = [
        {"weight": torch.tensor([1.0, 3.0]), "num_batches_tracked": torch.tensor(1)},
        {"weight": torch.tensor([3.0, 7.0]), "num_batches_tracked": torch.tensor(4)},
    ]
    aggregated = aggregate_state_dicts(states, client_sizes=[1, 3])
    assert torch.allclose(aggregated["weight"], torch.tensor([2.5, 6.0]))


def test_fedavg_copies_integer_buffers_from_largest_client() -> None:
    states = [
        {"weight": torch.tensor([1.0]), "num_batches_tracked": torch.tensor(1)},
        {"weight": torch.tensor([5.0]), "num_batches_tracked": torch.tensor(8)},
    ]
    aggregated = aggregate_state_dicts(states, client_sizes=[2, 5])
    assert aggregated["num_batches_tracked"].item() == 8

