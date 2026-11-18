import json
from pathlib import Path

import numpy as np
import torch

from src.models.transformer_pose_hands import (
    FEATURE_END,
    FEATURE_START,
    FULL_FEATURE_DIM,
    MODEL_FEATURE_DIM,
    SEQUENCE_LENGTH,
    TransformerClassifier,
    checkpoint_model,
    extract_model_features,
    load_label_mapping,
    normalize_features,
    sample_uniform_frames,
)


ROOT = Path(__file__).resolve().parents[1]


def test_label_mapping_is_contiguous_and_matches_metadata():
    mapping = load_label_mapping(
        ROOT / "data" / "metadata" / "label_mapping.json"
    )
    assert len(mapping) == 100
    assert set(mapping.values()) == set(range(100))

    for filename in (
        "train_signer_disjoint.csv",
        "validation_signer_disjoint.csv",
        "test_signer_disjoint.csv",
    ):
        lines = (ROOT / "data" / "metadata" / filename).read_text(
            encoding="utf-8"
        ).splitlines()[1:]
        glosses = {line.split(",", 3)[2] for line in lines}
        assert glosses <= set(mapping)


def test_feature_slice_and_uniform_sampling_match_offline_contract():
    raw = np.arange(60 * FULL_FEATURE_DIM, dtype=np.float32).reshape(
        60, FULL_FEATURE_DIM
    )
    selected = extract_model_features(raw)
    assert selected.shape == (60, MODEL_FEATURE_DIM)
    np.testing.assert_array_equal(
        selected[:, 0], raw[:, FEATURE_START]
    )
    np.testing.assert_array_equal(
        selected[:, -1], raw[:, FEATURE_END - 1]
    )

    sampled = sample_uniform_frames(selected)
    assert sampled.shape == (SEQUENCE_LENGTH, MODEL_FEATURE_DIM)
    np.testing.assert_array_equal(sampled[0], selected[0])
    np.testing.assert_array_equal(sampled[-1], selected[-1])


def test_normalization_supports_batch_and_rejects_invalid_shape():
    raw = np.ones((2, SEQUENCE_LENGTH, FULL_FEATURE_DIM), dtype=np.float32)
    normalized = normalize_features(
        raw,
        np.zeros(MODEL_FEATURE_DIM, dtype=np.float32),
        np.ones(MODEL_FEATURE_DIM, dtype=np.float32),
    )
    assert normalized.shape == (2, SEQUENCE_LENGTH, MODEL_FEATURE_DIM)
    np.testing.assert_array_equal(
        normalized, np.ones_like(normalized)
    )


def test_checkpoint_loads_with_expected_tensor_shape():
    checkpoint_path = (
        ROOT
        / "models"
        / "transformer_pose_hands"
        / "transformer_pose_hands.pt"
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = checkpoint_model(checkpoint)
    output = model(torch.zeros(2, SEQUENCE_LENGTH, MODEL_FEATURE_DIM))
    assert output.shape == (2, checkpoint["classes"])
