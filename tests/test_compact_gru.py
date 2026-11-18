import numpy as np
import pytest
import torch

from scripts.experiment_02_compact_gru import grouped_folds, sqrt_inverse_class_weights
from src.models.compact_gru import CompactGRUClassifier


def test_compact_gru_forward_has_expected_shape_and_finite_logits():
    model = CompactGRUClassifier(n_classes=5)
    inputs = torch.zeros((4, 30, 180), dtype=torch.float32)
    logits = model(inputs)
    assert logits.shape == (4, 5)
    assert torch.isfinite(logits).all()


def test_compact_gru_rejects_wrong_sequence_shape_and_nonfinite_input():
    model = CompactGRUClassifier(n_classes=5)
    with pytest.raises(ValueError, match="Expected input shape"):
        model(torch.zeros((2, 29, 180)))
    invalid = torch.zeros((2, 30, 180))
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        model(invalid)


def test_training_only_sqrt_inverse_class_weights_are_bounded_and_mean_one():
    labels = np.array([0] * 8 + [1] * 2 + [2] * 1, dtype=np.int64)
    weights = sqrt_inverse_class_weights(labels, n_classes=3)
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert float(weights.max()) <= 2.0 + 1e-6
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-6)
    assert weights[2] > weights[1] > weights[0]


def test_grouped_folds_keep_signer_groups_disjoint_and_class_covered():
    labels = np.tile(np.arange(3, dtype=np.int64), 6)
    groups = np.repeat([f"dataset::signer_{i}" for i in range(6)], 3)
    folds, n_splits = grouped_folds(labels, groups)
    assert n_splits == 3
    for fit_indices, score_indices in folds:
        assert set(groups[fit_indices]).isdisjoint(set(groups[score_indices]))
        assert set(labels[score_indices]) == {0, 1, 2}

