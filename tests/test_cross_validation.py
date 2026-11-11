import numpy as np
import pandas as pd
import pytest

from src.cross_validation import (
    create_fold_assignments,
    fit_normalization,
    validate_fold_assignments,
)


def development_fixture():
    rows = []
    for signer in range(10):
        for label in range(5):
            rows.append(
                {
                    "dataset": "fixture",
                    "signer_id": signer,
                    "gloss": f"class-{label}",
                    "keypoint_path": f"fixture-{signer}-{label}.npy",
                }
            )
    return pd.DataFrame(rows)


def test_signers_are_separated_and_each_example_has_one_fold():
    assignments, reports = create_fold_assignments(development_fixture())
    assert assignments["fold"].isin(range(1, 6)).all()
    assert assignments["example_id"].nunique() == len(assignments)
    assert all(item["signer_overlap_count"] == 0 for item in reports)
    for fold in range(1, 6):
        train = set(assignments.loc[assignments.fold != fold, "group_id"])
        validation = set(assignments.loc[assignments.fold == fold, "group_id"])
        assert train.isdisjoint(validation)


def test_validation_reports_missing_classes_instead_of_claiming_coverage():
    assignments = pd.DataFrame(
        {
            "fold": [1, 1, 2, 2],
            "gloss": ["a", "b", "a", "b"],
            "group_id": ["x", "y", "z", "w"],
        }
    )
    report = validate_fold_assignments(assignments, n_splits=2)
    assert report[0]["missing_validation_classes"] == []

    assignments.loc[assignments.fold == 1, "gloss"] = "a"
    report = validate_fold_assignments(assignments, n_splits=2)
    assert report[0]["missing_validation_classes"] == ["b"]


def test_normalization_is_fit_from_training_partition_only():
    training = np.ones((2, 30, 258), dtype=np.float32)
    validation = np.full((1, 30, 258), 100.0, dtype=np.float32)
    mean, std = fit_normalization(training)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(std, 1.0)
    assert np.all((validation - mean) / std == 99.0)


def test_overlap_is_rejected():
    assignments = pd.DataFrame(
        {
            "fold": [1, 2],
            "gloss": ["a", "a"],
            "group_id": ["same", "same"],
        }
    )
    with pytest.raises(AssertionError, match="signer overlap"):
        validate_fold_assignments(assignments, n_splits=2)


def test_fold_checkpoints_are_isolated_from_existing_model_paths():
    from scripts.cross_validate_transformer import CHECKPOINTS

    assert CHECKPOINTS.name == "checkpoints"
    assert "transformer_pose_hands.pt" not in {
        f.name for f in CHECKPOINTS.glob("*.pt")
    }
