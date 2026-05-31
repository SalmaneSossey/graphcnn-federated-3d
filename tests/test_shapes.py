"""Model shape tests."""

from __future__ import annotations

import torch

from src.models.common import PointCloudMLP, count_parameters
from src.models.pointgcn import PointGCN
from src.models.rscnn import RSCNN
from src.models.student import StudentPointCloudMLP


def test_placeholder_models_return_class_logits() -> None:
    batch_size = 4
    num_points = 1024
    input_channels = 6
    num_classes = 10
    points = torch.randn(batch_size, num_points, input_channels)

    for model in (
        PointGCN(num_classes=num_classes, input_channels=input_channels),
        RSCNN(num_classes=num_classes, input_channels=input_channels),
        PointCloudMLP(num_classes=num_classes, input_channels=input_channels),
    ):
        logits = model(points)
        assert logits.shape == (batch_size, num_classes)


def test_student_has_at_least_4x_fewer_parameters_than_teacher_scaffold() -> None:
    teacher = PointGCN(num_classes=10, input_channels=6, hidden_dim=128)
    student = StudentPointCloudMLP(num_classes=10, input_channels=6, hidden_dim=32)
    assert count_parameters(student) * 4 <= count_parameters(teacher)

