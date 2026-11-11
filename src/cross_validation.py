"""Signer-aware five-fold cross-validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


N_FOLDS = 5
RANDOM_STATE = 42
GROUP_COLUMN = "group_id"


def add_group_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    result[GROUP_COLUMN] = (
        result["dataset"].astype(str) + "::" + result["signer_id"].astype(str)
    )
    return result


def load_development_data(
    train_csv: Path, validation_csv: Path
) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    validation = pd.read_csv(validation_csv)
    required = {"dataset", "gloss", "signer_id", "keypoint_path"}
    for name, dataframe in (("train", train), ("validation", validation)):
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"{name} metadata is missing columns: {sorted(missing)}")
    development = pd.concat([train, validation], ignore_index=True)
    return add_group_ids(development)


def create_fold_assignments(
    development: pd.DataFrame,
    n_splits: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if n_splits != 5:
        raise ValueError("This workflow requires exactly five folds.")
    if GROUP_COLUMN not in development:
        development = add_group_ids(development)
    if development[GROUP_COLUMN].nunique() < n_splits:
        raise ValueError("There must be at least five signer groups.")

    assignments = development.copy()
    assignments.insert(0, "example_id", np.arange(len(assignments)))
    assignments["fold"] = -1
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for fold, (_, validation_indices) in enumerate(
        splitter.split(assignments, assignments["gloss"], assignments[GROUP_COLUMN]),
        start=1,
    ):
        assignments.loc[validation_indices, "fold"] = fold

    if (assignments["fold"] < 1).any():
        raise RuntimeError("Some examples were not assigned to a fold.")
    report = validate_fold_assignments(assignments, n_splits=n_splits)
    return assignments, report


def validate_fold_assignments(
    assignments: pd.DataFrame, n_splits: int = N_FOLDS
) -> list[dict[str, object]]:
    required = {"fold", "gloss", GROUP_COLUMN}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Fold assignments are missing columns: {sorted(missing)}")

    all_classes = sorted(assignments["gloss"].astype(str).unique())
    reports: list[dict[str, object]] = []
    for fold in range(1, n_splits + 1):
        validation = assignments[assignments["fold"] == fold]
        training = assignments[assignments["fold"] != fold]
        train_groups = set(training[GROUP_COLUMN])
        validation_groups = set(validation[GROUP_COLUMN])
        overlap = sorted(train_groups & validation_groups)
        if overlap:
            raise AssertionError(f"Fold {fold} has signer overlap: {overlap}")
        validation_counts = validation["gloss"].value_counts().to_dict()
        training_counts = training["gloss"].value_counts().to_dict()
        missing_validation = sorted(set(all_classes) - set(validation_counts))
        missing_training = sorted(set(all_classes) - set(training_counts))
        reports.append(
            {
                "fold": fold,
                "training_examples": int(len(training)),
                "validation_examples": int(len(validation)),
                "training_groups": int(training[GROUP_COLUMN].nunique()),
                "validation_groups": int(validation[GROUP_COLUMN].nunique()),
                "training_class_counts": {
                    str(key): int(value) for key, value in training_counts.items()
                },
                "validation_class_counts": {
                    str(key): int(value) for key, value in validation_counts.items()
                },
                "missing_training_classes": missing_training,
                "missing_validation_classes": missing_validation,
                "signer_overlap_count": 0,
            }
        )
    return reports


def fit_normalization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 3 or features.shape[-1] != 258:
        raise ValueError(f"Expected (examples, frames, 258), got {features.shape}")
    mean = features.reshape(-1, 258).mean(axis=0).astype(np.float32)
    std = features.reshape(-1, 258).std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def fold_indices(assignments: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    validation = assignments["fold"].to_numpy() == fold
    return np.flatnonzero(~validation), np.flatnonzero(validation)
