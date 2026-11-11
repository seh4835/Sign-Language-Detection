"""Shared preprocessing and model definitions for the controlled v2 run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


FRAME_COUNT = 30
RAW_FEATURE_DIM = 177
INPUT_DIM = 180
GROUP_SLICES = {
    "body_pose": slice(0, 51),
    "left_hand": slice(51, 114),
    "right_hand": slice(114, 177),
}
PRESENCE_INDEX = {"body_pose": 0, "left_hand": 1, "right_hand": 2}


@dataclass(frozen=True)
class FeatureNormalizer:
    """Mask-aware feature-wise standardization fitted on training sequences."""

    means: np.ndarray
    scales: np.ndarray
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.means.shape != (RAW_FEATURE_DIM,) or self.scales.shape != (RAW_FEATURE_DIM,):
            raise ValueError("Normalizer statistics must each have shape (177,).")
        if not np.isfinite(self.means).all() or not np.isfinite(self.scales).all():
            raise ValueError("Normalizer statistics must be finite.")
        if (self.scales < self.epsilon).any():
            raise ValueError("Normalizer scales must be at least epsilon.")

    @classmethod
    def fit(
        cls,
        sequences: np.ndarray,
        presence: np.ndarray,
        *,
        epsilon: float = 1e-6,
    ) -> "FeatureNormalizer":
        features = np.asarray(sequences, dtype=np.float64)
        indicators = np.asarray(presence, dtype=np.float64)
        if features.ndim != 3 or features.shape[2] != RAW_FEATURE_DIM:
            raise ValueError("Expected sequences with shape (N, 30, 177).")
        if indicators.shape != features.shape[:2] + (3,):
            raise ValueError("Presence must have shape (N, 30, 3).")
        if not np.isfinite(features).all() or not np.isfinite(indicators).all():
            raise ValueError("Training features and presence must be finite.")
        if not np.isin(indicators, (0.0, 1.0)).all():
            raise ValueError("Presence indicators must be binary.")

        means = np.zeros(RAW_FEATURE_DIM, dtype=np.float64)
        scales = np.ones(RAW_FEATURE_DIM, dtype=np.float64)
        for group, group_slice in GROUP_SLICES.items():
            detected = indicators[..., PRESENCE_INDEX[group]].astype(bool).reshape(-1)
            values = features[..., group_slice].reshape(-1, group_slice.stop - group_slice.start)
            if not detected.any():
                raise ValueError(f"No detected training frames for {group}.")
            detected_values = values[detected]
            means[group_slice] = detected_values.mean(axis=0)
            scales[group_slice] = np.maximum(detected_values.std(axis=0), epsilon)
        return cls(means.astype(np.float32), scales.astype(np.float32), epsilon)

    def transform(self, sequences: np.ndarray, presence: np.ndarray) -> np.ndarray:
        features = np.asarray(sequences, dtype=np.float32)
        indicators = np.asarray(presence, dtype=np.float32)
        if features.shape[-1] != RAW_FEATURE_DIM:
            raise ValueError("Expected the last feature dimension to be 177.")
        if indicators.shape != features.shape[:-1] + (3,):
            raise ValueError("Presence shape must match the sequence dimensions.")
        if not np.isfinite(features).all() or not np.isfinite(indicators).all():
            raise ValueError("Features and presence must be finite.")
        output = np.zeros_like(features, dtype=np.float32)
        for group, group_slice in GROUP_SLICES.items():
            detected = indicators[..., PRESENCE_INDEX[group]] == 1.0
            standardized = (
                features[..., group_slice] - self.means[group_slice]
            ) / self.scales[group_slice]
            output[..., group_slice] = np.where(
                detected[..., None], standardized, 0.0
            )
        if not np.isfinite(output).all():
            raise ValueError("Normalized features contain non-finite values.")
        return output

    def transform_lstm(self, sequences: np.ndarray, presence: np.ndarray) -> np.ndarray:
        normalized = self.transform(sequences, presence)
        indicators = np.asarray(presence, dtype=np.float32)
        return np.concatenate((normalized, indicators), axis=-1)

    def to_dict(self) -> dict[str, object]:
        return {
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "epsilon": self.epsilon,
            "continuous_feature_dim": RAW_FEATURE_DIM,
            "input_dim": INPUT_DIM,
            "normalized_groups": list(GROUP_SLICES),
            "presence_indicators_unchanged": True,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FeatureNormalizer":
        return cls(
            np.asarray(data["means"], dtype=np.float32),
            np.asarray(data["scales"], dtype=np.float32),
            float(data.get("epsilon", 1e-6)),
        )


class LSTMClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, 128, num_layers=2, batch_first=True, dropout=0.25)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1]))
