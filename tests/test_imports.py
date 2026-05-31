"""Import smoke tests."""

from __future__ import annotations


def test_core_imports() -> None:
    from src.data.dataset import ShapeNetPointCloudDataset
    from src.distillation.losses import distillation_loss
    from src.federated.fedavg import aggregate_state_dicts
    from src.models.pointgcn import PointGCN
    from src.models.rscnn import RSCNN

    assert ShapeNetPointCloudDataset is not None
    assert distillation_loss is not None
    assert aggregate_state_dicts is not None
    assert PointGCN is not None
    assert RSCNN is not None

