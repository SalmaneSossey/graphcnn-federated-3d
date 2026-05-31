"""Manual FedAvg aggregation for PyTorch state_dict objects."""

from __future__ import annotations

from collections.abc import Sequence

import torch


StateDict = dict[str, torch.Tensor]


def aggregate_state_dicts(
    client_states: Sequence[StateDict],
    client_sizes: Sequence[int],
) -> StateDict:
    """Aggregate client state_dicts using weighted FedAvg.

    Floating tensors are averaged by local dataset size. Non-floating tensors,
    such as BatchNorm counters, are copied from the largest client to avoid
    invalid fractional integer buffers.
    """
    if not client_states:
        raise ValueError("client_states must not be empty.")
    if len(client_states) != len(client_sizes):
        raise ValueError("client_states and client_sizes must have the same length.")
    if any(size < 0 for size in client_sizes):
        raise ValueError("client_sizes must be non-negative.")

    total_size = sum(client_sizes)
    if total_size <= 0:
        raise ValueError("At least one client must have a positive dataset size.")

    keys = client_states[0].keys()
    for state in client_states[1:]:
        if state.keys() != keys:
            raise ValueError("All client state_dicts must have identical keys.")

    largest_client_idx = max(range(len(client_sizes)), key=lambda idx: client_sizes[idx])
    aggregated: StateDict = {}

    for key in keys:
        first_tensor = client_states[0][key]
        if torch.is_floating_point(first_tensor) or torch.is_complex(first_tensor):
            avg = torch.zeros_like(first_tensor, dtype=first_tensor.dtype)
            for state, size in zip(client_states, client_sizes, strict=True):
                avg = avg + state[key].to(dtype=first_tensor.dtype) * (size / total_size)
            aggregated[key] = avg
        else:
            aggregated[key] = client_states[largest_client_idx][key].clone()
    return aggregated

