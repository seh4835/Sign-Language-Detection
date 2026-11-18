import numpy as np
import pytest

from src.models.yolo_pose_hands_lstm_v2 import FeatureNormalizer


def test_normalization_uses_detected_frames_and_preserves_missing_groups():
    features = np.zeros((1, 2, 177), dtype=np.float32)
    presence = np.zeros((1, 2, 3), dtype=np.float32)
    presence[0, 0] = [1, 1, 0]
    presence[0, 1, 2] = 1
    features[0, 0, :51] = 10
    features[0, 0, 51:114] = 4
    normalizer = FeatureNormalizer.fit(features, presence)

    sample = features.copy()
    sample[0, 1, :51] = 20
    sample[0, 1, 51:114] = 8
    transformed = normalizer.transform(sample, presence)
    assert np.count_nonzero(transformed[0, 1, :114]) == 0
    assert np.count_nonzero(transformed[0, 0, 114:]) == 0
    assert np.allclose(transformed[0, 0, :51], 10 - 10)
    assert np.allclose(transformed[0, 0, 51:114], 4 - 4)


def test_normalization_is_numerically_stable_for_constant_detected_features():
    features = np.ones((2, 3, 177), dtype=np.float32)
    presence = np.ones((2, 3, 3), dtype=np.float32)
    normalizer = FeatureNormalizer.fit(features, presence)
    result = normalizer.transform_lstm(features, presence)
    assert np.isfinite(result).all()
    assert np.allclose(result[:, :, :177], 0)
    np.testing.assert_array_equal(result[:, :, 177:], presence)


def test_normalizer_rejects_groups_without_training_detections():
    features = np.zeros((1, 1, 177), dtype=np.float32)
    presence = np.ones((1, 1, 3), dtype=np.float32)
    presence[0, 0, 2] = 0
    with pytest.raises(ValueError, match="right_hand"):
        FeatureNormalizer.fit(features, presence)
