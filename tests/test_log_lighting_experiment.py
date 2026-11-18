import csv

import pytest

from scripts.log_lighting_experiment import (
    FIELD_NAMES,
    append_experiment_record,
    validate_experiment_ids,
)


def _record(experiment_id="EXP01"):
    return {
        "experiment_id": experiment_id,
        "participant_id": "P01",
        "lighting_condition": "condition-a",
        "mean_brightness": 100.0,
        "median_brightness": 98.0,
        "std_brightness": 12.5,
        "percent_below_50": 4.0,
        "percent_above_200": 1.5,
        "predicted_sign": "hello",
        "actual_sign": "hello",
    }


def test_append_creates_directory_header_and_record(tmp_path):
    output_path = tmp_path / "reports" / "fairness" / "results.csv"

    append_experiment_record(_record(), output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert output_path.parent.is_dir()
    assert list(rows[0]) == list(FIELD_NAMES)
    assert rows[0]["experiment_id"] == "EXP01"
    assert rows[0]["mean_brightness"] == "100.0"


def test_append_adds_records_without_overwriting(tmp_path):
    output_path = tmp_path / "results.csv"

    append_experiment_record(_record("EXP01"), output_path)
    append_experiment_record(_record("EXP02"), output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in rows] == ["EXP01", "EXP02"]


@pytest.mark.parametrize(
    "participant_id,experiment_id",
    [("P01", "EXP01"), ("P99", "EXP99")],
)
def test_validate_experiment_ids_accepts_documented_formats(
    participant_id, experiment_id
):
    assert validate_experiment_ids(participant_id, experiment_id) == (
        participant_id,
        experiment_id,
    )


@pytest.mark.parametrize(
    "participant_id,experiment_id",
    [
        ("P00", "EXP01"),
        ("P1", "EXP01"),
        ("P100", "EXP01"),
        ("P01", "EXP00"),
        ("P01", "EXP1"),
        ("P01", "EXP100"),
        ("Jane Doe", "EXP01"),
    ],
)
def test_validate_experiment_ids_rejects_invalid_formats(
    participant_id, experiment_id
):
    with pytest.raises(ValueError):
        validate_experiment_ids(participant_id, experiment_id)


@pytest.mark.parametrize(
    "field,value",
    [("participant_id", "P1"), ("experiment_id", "E01")],
)
def test_logger_rejects_invalid_ids(tmp_path, field, value):
    record = _record()
    record[field] = value

    with pytest.raises(ValueError, match="must match the format"):
        append_experiment_record(record, tmp_path / "results.csv")


@pytest.mark.parametrize(
    "change",
    [
        {"actual_sign": ""},
        {"mean_brightness": "not-a-number"},
        {"std_brightness": float("nan")},
        {"raw_frame": b"not-accepted"},
    ],
)
def test_append_rejects_invalid_or_non_scalar_fields(tmp_path, change):
    record = _record()
    record.update(change)

    with pytest.raises(ValueError):
        append_experiment_record(record, tmp_path / "results.csv")
