"""Append one validated lighting experiment record to a CSV file."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / "fairness" / "lighting_experiment_results.csv"

FIELD_NAMES = (
    "experiment_id",
    "participant_id",
    "lighting_condition",
    "mean_brightness",
    "median_brightness",
    "std_brightness",
    "percent_below_50",
    "percent_above_200",
    "predicted_sign",
    "actual_sign",
)

NUMERIC_FIELDS = {
    "mean_brightness",
    "median_brightness",
    "std_brightness",
    "percent_below_50",
    "percent_above_200",
}

ID_PATTERNS = {
    "participant_id": re.compile(r"P(?:0[1-9]|[1-9][0-9])"),
    "experiment_id": re.compile(r"EXP(?:0[1-9]|[1-9][0-9])"),
}


def validate_experiment_ids(
    participant_id: str,
    experiment_id: str,
) -> tuple[str, str]:
    """Validate and return IDs in the anonymous experiment-code formats."""
    for field, value in (
        ("participant_id", participant_id),
        ("experiment_id", experiment_id),
    ):
        if not isinstance(value, str) or not ID_PATTERNS[field].fullmatch(value):
            expected = "P01" if field == "participant_id" else "EXP01"
            raise ValueError(f"{field} must match the format {expected}.")
    return participant_id, experiment_id


def _validate_record(record: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping.")

    missing = [field for field in FIELD_NAMES if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}.")

    unexpected = sorted(set(record) - set(FIELD_NAMES))
    if unexpected:
        raise ValueError(f"Unexpected fields are not allowed: {unexpected}.")

    validated: dict[str, object] = {}
    for field in FIELD_NAMES:
        value = record[field]
        if field in NUMERIC_FIELDS:
            if isinstance(value, bool):
                raise ValueError(f"{field} must be numeric.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field} must be numeric.") from error
            if not math.isfinite(numeric_value):
                raise ValueError(f"{field} must be finite.")
            validated[field] = numeric_value
        else:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string.")
            validated[field] = value
    validate_experiment_ids(
        str(validated["participant_id"]),
        str(validated["experiment_id"]),
    )
    return validated


def append_experiment_record(
    record: Mapping[str, object],
    output_path: Path = RESULTS_PATH,
) -> None:
    """Validate and append exactly one scalar experiment record to CSV."""
    validated = _validate_record(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_NAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(validated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in FIELD_NAMES:
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    record = {field: getattr(args, field) for field in FIELD_NAMES}
    append_experiment_record(record, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
