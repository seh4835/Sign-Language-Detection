import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.temporal_baseline import (
    load_artifact,
    predict_sequence,
    sample_uniform_sequence,
    save_artifact,
    summary_features,
)
from src.models.yolo_pose_hands_lstm_v2 import FeatureNormalizer


def make_normalizer():
    features = np.zeros((4, 30, 177), dtype=np.float32)
    presence = np.ones((4, 30, 3), dtype=np.float32)
    features[..., 0] = np.arange(120, dtype=np.float32).reshape(4, 30)
    return FeatureNormalizer.fit(features, presence)


def test_summary_features_has_exact_360_dimension():
    sequence = np.zeros((30, 180), dtype=np.float32)
    assert summary_features(sequence).shape == (1, 360)
    assert summary_features(np.zeros((3, 30, 180), dtype=np.float32)).shape == (3, 360)


def test_uniform_sampling_short_and_long_intervals():
    with np.testing.assert_raises(ValueError):
        sample_uniform_sequence(np.zeros((29, 180), dtype=np.float32))
    sequence = np.arange(60 * 180, dtype=np.float32).reshape(60, 180)
    sampled = sample_uniform_sequence(sequence)
    np.testing.assert_array_equal(
        sampled[:, 0], np.linspace(0, 59, 30, dtype=np.int64) * 180
    )


def test_artifact_loading_and_prediction_round_trip(tmp_path):
    normalizer = make_normalizer()
    rng = np.random.default_rng(42)
    train_x = rng.normal(size=(12, 360)).astype(np.float32)
    train_y = np.arange(12) % 2
    classifier = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    classifier.fit(train_x, train_y)
    mapping = {"again": 0, "all": 1}
    save_artifact(tmp_path, classifier, normalizer, mapping | {
        str(index): index for index in range(2, 100)
    })
    loaded, loaded_normalizer, labels = load_artifact(tmp_path)
    assert len(labels) == 100
    label, confidence = predict_sequence(
        loaded, loaded_normalizer, np.zeros((30, 180), dtype=np.float32)
    )
    assert label in (0, 1)
    assert 0.0 <= confidence <= 1.0
