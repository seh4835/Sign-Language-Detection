"""Run a deterministic, in-memory temporal baseline smoke test for CI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.temporal_baseline import summary_features  # noqa: E402
from src.models.yolo_pose_hands_lstm_v2 import FeatureNormalizer  # noqa: E402


RANDOM_SEED = 42
FRAME_COUNT = 30
RAW_FEATURE_DIM = 177
LSTM_FEATURE_DIM = 180
CLASS_COUNT = 4
TRAIN_EXAMPLES_PER_CLASS = 8
VALIDATION_EXAMPLES_PER_CLASS = 2


def make_synthetic_sequences(
    examples_per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.repeat(np.arange(CLASS_COUNT), examples_per_class)
    sequences = rng.normal(
        0.0,
        0.02,
        size=(len(labels), FRAME_COUNT, RAW_FEATURE_DIM),
    ).astype(np.float32)
    sequences[:, :, 0] += labels[:, None].astype(np.float32)
    presence = np.ones(
        (len(labels), FRAME_COUNT, LSTM_FEATURE_DIM - RAW_FEATURE_DIM),
        dtype=np.float32,
    )
    return sequences, presence, labels


def main() -> int:
    rng = np.random.default_rng(RANDOM_SEED)
    train_features, train_presence, train_labels = make_synthetic_sequences(
        TRAIN_EXAMPLES_PER_CLASS, rng
    )
    validation_features, validation_presence, validation_labels = (
        make_synthetic_sequences(VALIDATION_EXAMPLES_PER_CLASS, rng)
    )

    normalizer = FeatureNormalizer.fit(train_features, train_presence)
    train_normalized = normalizer.transform_lstm(train_features, train_presence)
    train_summary = summary_features(train_normalized)

    validation_normalized = normalizer.transform_lstm(
        validation_features, validation_presence
    )
    validation_summary = summary_features(validation_normalized)

    classifier = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    classifier.fit(train_summary, train_labels)
    predictions = classifier.predict(validation_summary)
    accuracy = float(accuracy_score(validation_labels, predictions))
    print(f"Synthetic validation accuracy: {accuracy:.4f}")
    print("CI smoke training succeeded: in-memory pipeline completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
