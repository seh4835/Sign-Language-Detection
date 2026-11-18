import numpy as np
import pytest

from scripts.experiment_01_temporal_bin_velocity import make_grouped_folds
from src.models.temporal_bin_velocity import (
    frame_to_frame_velocity,
    temporal_bin_velocity_features,
)


def make_sequence():
    sequence = np.zeros((30, 180), dtype=np.float32)
    sequence[:, 177:] = 1.0
    sequence[:, 0] = np.arange(30, dtype=np.float32)
    return sequence


def test_temporal_bin_velocity_shape_order_and_finite_values():
    sequence = make_sequence()
    features = temporal_bin_velocity_features(sequence)

    assert features.shape == (2160,)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(features[:1080].reshape(6, 180)[:, 0], [2, 7, 12, 17, 22, 27])
    np.testing.assert_allclose(features[1080:].reshape(6, 180)[:, 0], np.ones(6))


def test_velocity_masks_continuous_deltas_when_a_landmark_group_is_missing():
    sequence = make_sequence()
    sequence[4, 51:114] = 0.0
    sequence[4, 178] = 0.0
    sequence[5, 51:114] = 10.0
    sequence[5, 178] = 1.0
    velocity = frame_to_frame_velocity(sequence)

    assert velocity.shape == (29, 180)
    # The missing-to-present jump is not interpreted as physical hand velocity.
    assert np.all(velocity[4, 51:114] == 0.0)
    # Presence flag deltas remain explicit.
    assert velocity[4, 178] == 1.0
    # A continuously detected body point retains its signed delta.
    assert velocity[0, 0] == 1.0


def test_temporal_features_reject_nonfinite_values_and_nonbinary_presence():
    sequence = make_sequence()
    sequence[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        temporal_bin_velocity_features(sequence)

    sequence = make_sequence()
    sequence[0, 177] = 0.5
    with pytest.raises(ValueError, match="binary"):
        temporal_bin_velocity_features(sequence)


def test_temporal_features_support_batches():
    batch = np.stack([make_sequence(), make_sequence()])
    result = temporal_bin_velocity_features(batch)
    assert result.shape == (2, 2160)
    assert np.isfinite(result).all()


def test_grouped_cv_keeps_signer_groups_separate_and_all_classes_present():
    labels = np.tile(np.arange(3, dtype=np.int64), 6)
    groups = np.repeat([f"dataset::signer_{i}" for i in range(6)], 3)
    folds, n_splits = make_grouped_folds(labels, groups)

    assert n_splits == 3
    for train_idx, validation_idx in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[validation_idx]))
        assert set(labels[validation_idx]) == {0, 1, 2}

