"""Generate a development-only Evidently keypoint feature drift report."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.transformer_pose_hands import extract_model_features


ROOT = Path(__file__).resolve().parents[1]
TRAIN_METADATA = ROOT / "data" / "metadata" / "train_signer_disjoint.csv"
VALIDATION_METADATA = ROOT / "data" / "metadata" / "validation_signer_disjoint.csv"
REPORT_DIRECTORY = ROOT / "reports" / "drift"
HTML_OUTPUT = REPORT_DIRECTORY / "keypoint_drift_report.html"
JSON_OUTPUT = REPORT_DIRECTORY / "keypoint_drift_summary.json"

EXPECTED_KEYPOINT_SHAPE = (30, 1662)
MODEL_FEATURE_COUNT = 258
STATISTICS = ("mean", "std", "min", "max")


def load_metadata(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if "keypoint_path" not in dataframe.columns:
        raise ValueError(f"{path} is missing the required keypoint_path column.")
    # The split column preserves the original dataset split; the selected
    # signer-disjoint CSV determines the current development partition.
    return dataframe


def feature_columns() -> list[str]:
    return [
        f"feature_{feature_index:03d}_{statistic}"
        for feature_index in range(MODEL_FEATURE_COUNT)
        for statistic in STATISTICS
    ]


def summarize_feature_statistics(features: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            features.mean(axis=0),
            features.std(axis=0),
            features.min(axis=0),
            features.max(axis=0),
        )
    ).reshape(-1)


def summarize_keypoints(
    dataframe: pd.DataFrame,
    metadata_path: Path,
) -> pd.DataFrame:
    rows: list[np.ndarray] = []
    for row_number, (_, row) in enumerate(dataframe.iterrows(), start=2):
        keypoint_path = Path(str(row["keypoint_path"]))
        if not keypoint_path.is_absolute():
            keypoint_path = ROOT / keypoint_path
        if not keypoint_path.is_file():
            raise FileNotFoundError(
                f"Missing keypoint file referenced by {metadata_path} row "
                f"{row_number}: {keypoint_path}"
            )

        try:
            array = np.load(keypoint_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Could not load keypoint file from {metadata_path} row "
                f"{row_number}: {keypoint_path}"
            ) from error

        if array.shape != EXPECTED_KEYPOINT_SHAPE:
            raise ValueError(
                f"Malformed keypoint array from {metadata_path} row {row_number}: "
                f"{keypoint_path} has shape {array.shape}, expected "
                f"{EXPECTED_KEYPOINT_SHAPE}."
            )
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError as error:
            raise ValueError(
                f"Malformed non-numeric keypoint array from {metadata_path} "
                f"row {row_number}: {keypoint_path}"
            ) from error
        if not finite:
            raise ValueError(
                f"Keypoint array from {metadata_path} row {row_number} contains "
                f"non-finite values: {keypoint_path}"
            )

        features = extract_model_features(array)
        statistics = summarize_feature_statistics(features)
        rows.append(statistics)

    return pd.DataFrame(np.asarray(rows), columns=feature_columns())


def snapshot_as_dict(snapshot: Any) -> dict[str, Any] | None:
    as_dict = getattr(snapshot, "dict", None)
    if callable(as_dict):
        result = as_dict()
        if isinstance(result, dict):
            return result

    as_json = getattr(snapshot, "json", None)
    if callable(as_json):
        result = json.loads(as_json())
        if isinstance(result, dict):
            return result

    return None


def overall_drift_result(results: Any) -> bool | None:
    if not isinstance(results, dict):
        return None

    metrics = results.get("metrics")
    if not isinstance(metrics, list):
        return None

    for metric in metrics:
        if not isinstance(metric, dict):
            continue

        config = metric.get("config")
        value = metric.get("value")

        if not isinstance(config, dict) or not isinstance(value, dict):
            continue

        if config.get("type") != "evidently:metric_v2:DriftedColumnsCount":
            continue

        drift_share = config.get("drift_share")
        observed_share = value.get("share")

        if isinstance(drift_share, (int, float)) and isinstance(
            observed_share, (int, float)
        ):
            return observed_share > drift_share

    return None


def ensure_outputs_are_new() -> None:
    existing = [str(path) for path in (HTML_OUTPUT, JSON_OUTPUT) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing drift report output(s):\n"
            + "\n".join(existing)
        )


def main() -> None:
    ensure_outputs_are_new()
    train_metadata = load_metadata(TRAIN_METADATA)
    validation_metadata = load_metadata(VALIDATION_METADATA)

    reference_data = summarize_keypoints(train_metadata, TRAIN_METADATA)
    current_data = summarize_keypoints(validation_metadata, VALIDATION_METADATA)

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(HTML_OUTPUT))

    drift_results = snapshot_as_dict(snapshot)
    summary = {
        "reference_rows": len(reference_data),
        "current_rows": len(current_data),
        "monitoring_features": len(reference_data.columns),
        "overall_drift": overall_drift_result(drift_results),
        "drift_results": drift_results,
    }
    JSON_OUTPUT.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Reference sequences processed: {len(reference_data)}")
    print(f"Current sequences processed: {len(current_data)}")
    print(f"Monitoring features: {len(reference_data.columns)}")
    print(f"HTML report: {HTML_OUTPUT}")
    print(f"JSON summary: {JSON_OUTPUT}")
    print(f"Overall drift result: {summary['overall_drift']}")


if __name__ == "__main__":
    main()
