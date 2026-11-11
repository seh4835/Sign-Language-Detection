"""Shared Transformer model and preprocessing for pose-and-hand features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


SEQUENCE_LENGTH = 30
FULL_FEATURE_DIM = 1662
FEATURE_START = 1404
FEATURE_END = 1662
MODEL_FEATURE_DIM = FEATURE_END - FEATURE_START


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = SEQUENCE_LENGTH):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Expected at most {self.pe.size(1)} frames, got {x.size(1)}."
            )
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = MODEL_FEATURE_DIM,
        dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        classes: int = 100,
        dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Linear(input_dim, dim)
        self.position = PositionalEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=256,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=layers
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(dim, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(2) != self.projection.in_features:
            raise ValueError(
                "Expected input with shape (batch, frames, "
                f"{self.projection.in_features}), got {tuple(x.shape)}."
            )
        x = self.projection(x)
        x = self.position(x)
        x = self.encoder(x)
        return self.head(self.norm(x.mean(dim=1)))


def load_label_mapping(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as file:
        mapping = json.load(file)
    if not mapping or set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Label mapping must contain contiguous indices from 0.")
    return {str(label): int(index) for label, index in mapping.items()}


def extract_model_features(sequence: np.ndarray) -> np.ndarray:
    """Select the 258 pose-and-hand features used by the original model."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != FULL_FEATURE_DIM:
        raise ValueError(
            f"Expected (frames, {FULL_FEATURE_DIM}) features, got {array.shape}."
        )
    return array[:, FEATURE_START:FEATURE_END].copy()


def sample_uniform_frames(sequence: np.ndarray) -> np.ndarray:
    """Sample exactly 30 frames like the offline video extractor."""
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D sequence, got {array.shape}.")
    if len(array) < SEQUENCE_LENGTH:
        raise ValueError(
            f"At least {SEQUENCE_LENGTH} frames are required, got {len(array)}."
        )
    indices = np.linspace(
        0, len(array) - 1, SEQUENCE_LENGTH, dtype=int
    )
    return array[indices]


def checkpoint_model(
    checkpoint: dict[str, Any],
    *,
    dropout: float = 0.2,
    head_dropout: float = 0.2,
) -> TransformerClassifier:
    required = ("model_state_dict", "input_dim", "dim", "heads", "layers", "classes")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    if checkpoint["input_dim"] != MODEL_FEATURE_DIM:
        raise ValueError(
            f"Expected {MODEL_FEATURE_DIM} input features, "
            f"got {checkpoint['input_dim']}."
        )
    model = TransformerClassifier(
        input_dim=int(checkpoint["input_dim"]),
        dim=int(checkpoint["dim"]),
        heads=int(checkpoint["heads"]),
        layers=int(checkpoint["layers"]),
        classes=int(checkpoint["classes"]),
        dropout=dropout,
        head_dropout=head_dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_features(
    sequence: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D sequence, got {array.shape}.")
    if array.shape[-1] == FULL_FEATURE_DIM:
        features = array[..., FEATURE_START:FEATURE_END].copy()
    else:
        features = array
    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    std_array = np.asarray(std, dtype=np.float32).reshape(-1)
    if features.shape[-1] != MODEL_FEATURE_DIM:
        raise ValueError(
            f"Expected {MODEL_FEATURE_DIM} features, got {features.shape[-1]}."
        )
    if mean_array.shape != (MODEL_FEATURE_DIM,) or std_array.shape != (MODEL_FEATURE_DIM,):
        raise ValueError("Normalization statistics must each have shape (258,).")
    safe_std = np.where(std_array < 1e-6, 1.0, std_array)
    normalized = (features - mean_array) / safe_std
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized features contain non-finite values.")
    return normalized.astype(np.float32)
