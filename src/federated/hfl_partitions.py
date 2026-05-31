"""Client partition helpers for horizontal federated learning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_iid_partitions(dataframe: pd.DataFrame, num_clients: int, seed: int = 42) -> list[pd.DataFrame]:
    """Split rows randomly and evenly across clients."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return [part.reset_index(drop=True) for part in np.array_split(shuffled, num_clients)]


def make_non_iid_label_partitions(
    dataframe: pd.DataFrame,
    label_column: str,
    num_clients: int,
    classes_per_client: int = 2,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Create simple label-skewed non-IID client partitions."""
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if classes_per_client <= 0:
        raise ValueError("classes_per_client must be positive.")
    rng = np.random.default_rng(seed)
    labels = list(dataframe[label_column].dropna().unique())
    rng.shuffle(labels)
    client_parts = [[] for _ in range(num_clients)]

    client_label_sets = [
        {labels[(client_id * classes_per_client + offset) % len(labels)] for offset in range(classes_per_client)}
        for client_id in range(num_clients)
    ] if labels else [set() for _ in range(num_clients)]

    for label in labels:
        eligible_clients = [
            client_id for client_id, label_set in enumerate(client_label_sets) if label in label_set
        ]
        if not eligible_clients:
            eligible_clients = [len(client_parts) % num_clients]
        group = dataframe[dataframe[label_column] == label].sample(frac=1.0, random_state=seed)
        for split, client_id in zip(np.array_split(group, len(eligible_clients)), eligible_clients, strict=True):
            client_parts[client_id].append(split)

    partitions: list[pd.DataFrame] = []
    for client_id, groups in enumerate(client_parts):
        if not groups:
            partitions.append(dataframe.iloc[0:0].copy())
            continue
        merged = pd.concat(groups).sample(frac=1.0, random_state=seed + client_id)
        partitions.append(merged.reset_index(drop=True))
    return partitions
