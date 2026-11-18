"""Order-preserving temporal-bin and landmark-velocity features."""
from __future__ import annotations

import numpy as np


FRAME_COUNT = 30
FEATURE_DIM = 180
CONTINUOUS_DIM = 177
PRESENCE_DIM = 3
PRESENCE_SLICES = (slice(0, 51), slice(51, 114), slice(114, 177))


def _as_sequences(sequences: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(sequences, dtype=np.float32)
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1:] != (FRAME_COUNT, FEATURE_DIM):
        raise ValueError("Expected sequence shape (30, 180) or (N, 30, 180).")
    if not np.isfinite(values).all():
        raise ValueError("Sequences must contain only finite values.")
    presence = values[..., CONTINUOUS_DIM:]
    if not np.isin(presence, (0.0, 1.0)).all():
        raise ValueError("Presence indicators must be binary values.")
    return values, single


def frame_to_frame_velocity(sequence: np.ndarray) -> np.ndarray:
    """Return 29 signed frame differences with missing-group transitions masked.

    Continuous body/hand deltas are zero unless that feature group is detected
    in both adjacent frames. Presence-flag deltas are retained so detection
    appearance/disappearance remains represented explicitly.
    """
    values, single = _as_sequences(sequence)
    if not single:
        raise ValueError("frame_to_frame_velocity accepts one (30, 180) sequence.")
    deltas = values[:, 1:, :] - values[:, :-1, :]
    for presence_index, group_slice in enumerate(PRESENCE_SLICES):
        present_both = (
            (values[:, :-1, CONTINUOUS_DIM + presence_index] == 1.0)
            & (values[:, 1:, CONTINUOUS_DIM + presence_index] == 1.0)
        )
        deltas[:, :, group_slice] = np.where(
            present_both[:, :, None], deltas[:, :, group_slice], 0.0
        )
    result = deltas[0]
    if result.shape != (FRAME_COUNT - 1, FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError("Velocity calculation produced an invalid result.")
    return result


def temporal_bin_velocity_features(
    sequences: np.ndarray, bins: int = 6
) -> np.ndarray:
    """Return ordered bin means followed by ordered, pooled frame deltas.

    With 30 frames and six bins the output has 6*180 mean features and
    6*180 velocity features, for 2,160 values per sequence. A delta belongs
    to the bin containing its later frame. Works on one sequence or a batch.
    """
    values, single = _as_sequences(sequences)
    if not isinstance(bins, int) or bins < 1 or bins > FRAME_COUNT:
        raise ValueError(f"bins must be an integer from 1 to {FRAME_COUNT}.")
    edges = np.linspace(0, FRAME_COUNT, bins + 1, dtype=np.int64)
    if np.any(np.diff(edges) == 0):
        raise ValueError("Each temporal bin must contain at least one frame.")

    pooled_means = []
    for start, stop in zip(edges[:-1], edges[1:]):
        pooled_means.append(values[:, start:stop, :].mean(axis=1))
    bin_means = np.stack(pooled_means, axis=1)

    deltas = values[:, 1:, :] - values[:, :-1, :]
    for presence_index, group_slice in enumerate(PRESENCE_SLICES):
        present_both = (
            (values[:, :-1, CONTINUOUS_DIM + presence_index] == 1.0)
            & (values[:, 1:, CONTINUOUS_DIM + presence_index] == 1.0)
        )
        deltas[:, :, group_slice] = np.where(
            present_both[:, :, None], deltas[:, :, group_slice], 0.0
        )

    later_frames = np.arange(1, FRAME_COUNT)
    later_bins = np.searchsorted(edges[1:], later_frames, side="right")
    if later_bins.min() < 0 or later_bins.max() >= bins:
        raise ValueError("Could not assign all frame deltas to temporal bins.")
    pooled_velocity = []
    for bin_index in range(bins):
        in_bin = later_bins == bin_index
        if not in_bin.any():
            pooled_velocity.append(np.zeros((len(values), FEATURE_DIM), dtype=np.float32))
        else:
            pooled_velocity.append(deltas[:, in_bin, :].mean(axis=1))
    bin_velocity = np.stack(pooled_velocity, axis=1)

    features = np.concatenate((bin_means.reshape(len(values), -1),
                               bin_velocity.reshape(len(values), -1)), axis=1)
    expected_dim = 2 * bins * FEATURE_DIM
    if features.shape != (len(values), expected_dim) or not np.isfinite(features).all():
        raise ValueError("Temporal-bin features have an invalid shape or non-finite values.")
    return features[0] if single else features
