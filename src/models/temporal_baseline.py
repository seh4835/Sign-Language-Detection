"""Temporal mean/variance baseline preprocessing and artifact loading."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.models.yolo_pose_hands_lstm_v2 import FeatureNormalizer


FRAME_COUNT = 30
LSTM_FEATURE_DIM = 180
SUMMARY_FEATURE_DIM = 360


def summary_features(sequence: np.ndarray) -> np.ndarray:
    """Convert one or more normalized sequences to mean/std features."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (FRAME_COUNT, LSTM_FEATURE_DIM):
        raise ValueError("Expected sequence shape (30, 180) or (N, 30, 180).")
    if not np.isfinite(values).all():
        raise ValueError("Sequence contains non-finite values.")
    features = np.concatenate((values.mean(axis=1), values.std(axis=1)), axis=1)
    if features.shape[1] != SUMMARY_FEATURE_DIM:
        raise ValueError("Expected summary feature shape (N, 360).")
    return features


def sample_uniform_sequence(sequence: np.ndarray, count: int = FRAME_COUNT) -> np.ndarray:
    """Sample exactly 30 chronological frames across a captured interval."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != LSTM_FEATURE_DIM:
        raise ValueError("Expected captured sequence shape (N, 180).")
    if len(values) < count:
        raise ValueError(f"Need at least {count} captured frames, got {len(values)}.")
    indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[indices]


def save_artifact(
    directory: Path,
    classifier: object,
    normalizer: FeatureNormalizer,
    label_mapping: dict[str, int],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "classifier.pkl").open("wb") as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (directory / "normalization.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8"
    )
    (directory / "label_mapping.json").write_text(
        json.dumps(label_mapping, indent=2), encoding="utf-8"
    )


def load_artifact(directory: Path) -> tuple[object, FeatureNormalizer, list[str]]:
    with (directory / "classifier.pkl").open("rb") as handle:
        classifier = pickle.load(handle)
    normalizer = FeatureNormalizer.from_dict(
        json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
    )
    mapping = json.loads((directory / "label_mapping.json").read_text(encoding="utf-8"))
    labels = [gloss for gloss, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    if len(labels) != 100:
        raise ValueError("Artifact label mapping must contain 100 classes.")
    if getattr(classifier, "n_features_in_", None) != SUMMARY_FEATURE_DIM:
        raise ValueError("Classifier artifact must accept 360 summary features.")
    return classifier, normalizer, labels


def predict_sequence(
    classifier: object,
    normalizer: FeatureNormalizer,
    sequence: np.ndarray,
) -> tuple[int, float]:
    values = np.asarray(sequence, dtype=np.float32)
    if values.shape != (FRAME_COUNT, LSTM_FEATURE_DIM):
        raise ValueError("Prediction requires a (30, 180) sequence.")
    normalized = normalizer.transform_lstm(values[:, :177], values[:, 177:])
    features = summary_features(normalized)
    predicted = int(classifier.predict(features)[0])
    probabilities = classifier.predict_proba(features)[0]
    return predicted, float(probabilities[predicted])
