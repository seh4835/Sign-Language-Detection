"""Import historical Sign Language Recognition results into local MLflow."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Sign Language Recognition"

COMPARISON_FILE = ROOT / "reports" / "graphs" / "validation_model_comparison.csv"
BASELINE_FILE = ROOT / "reports" / "cross_validation" / "summary.csv"
WEIGHT_DECAY_FILE = (
    ROOT / "reports" / "cross_validation_weight_decay_1e2" / "summary.csv"
)
WEIGHT_DECAY_CONFIG_FILE = (
    ROOT / "reports" / "cross_validation_weight_decay_1e2" / "experiment_config.json"
)

COMMON_TAGS = {
    "architecture": "Transformer",
    "run_origin": "historical_results",
}

COMPARISON_REQUIRED_COLUMNS = {
    "model",
    "split",
    "top1_accuracy",
    "top5_accuracy",
    "examples",
    "checkpoint",
}
BASELINE_REQUIRED_COLUMNS = {
    "fold",
    "epochs_trained",
    "best_validation_loss",
    "validation_loss",
    "top1_accuracy",
    "top5_accuracy",
    "checkpoint",
}
WEIGHT_DECAY_REQUIRED_COLUMNS = {
    "fold",
    "epochs_trained",
    "best_validation_loss",
    "validation_loss",
    "top1_accuracy",
    "top5_accuracy",
    "training_loss_at_best_validation_loss",
    "training_accuracy_at_best_validation_loss",
    "validation_top1_at_best_validation_loss",
    "checkpoint",
}
CONFIG_HYPERPARAMETERS = (
    "weight_decay",
    "learning_rate",
    "batch_size",
    "max_epochs",
    "early_stopping_patience",
)
METRIC_COLUMNS = (
    "top1_accuracy",
    "top5_accuracy",
    "validation_loss",
    "best_validation_loss",
    "epochs_trained",
    "training_loss_at_best_validation_loss",
    "training_accuracy_at_best_validation_loss",
    "validation_top1_at_best_validation_loss",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def validate_csv_columns(path: Path, required_columns: set[str]) -> None:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        actual_columns = set(reader.fieldnames or [])
    missing_columns = sorted(required_columns - actual_columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing_columns)}"
        )


def validate_numeric_metrics(
    rows: list[dict[str, str]], source_name: str
) -> None:
    for row_number, row in enumerate(rows, start=1):
        for column in METRIC_COLUMNS:
            value = row.get(column, "").strip()
            if not value:
                continue
            try:
                numeric_value = float(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid numeric metric in {source_name} row {row_number}: "
                    f"{column}={value!r}"
                ) from error
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Non-finite metric in {source_name} row {row_number}: "
                    f"{column}={value!r}"
                )


def require_files() -> None:
    required = (
        COMPARISON_FILE,
        BASELINE_FILE,
        WEIGHT_DECAY_FILE,
        WEIGHT_DECAY_CONFIG_FILE,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required historical result file(s) are missing:\n"
            + "\n".join(missing)
        )


def load_inputs() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    validate_csv_columns(COMPARISON_FILE, COMPARISON_REQUIRED_COLUMNS)
    validate_csv_columns(BASELINE_FILE, BASELINE_REQUIRED_COLUMNS)
    validate_csv_columns(WEIGHT_DECAY_FILE, WEIGHT_DECAY_REQUIRED_COLUMNS)

    comparison_rows = read_csv_rows(COMPARISON_FILE)
    baseline_rows = read_csv_rows(BASELINE_FILE)
    weight_decay_rows = read_csv_rows(WEIGHT_DECAY_FILE)
    with WEIGHT_DECAY_CONFIG_FILE.open(encoding="utf-8") as file:
        weight_decay_config = json.load(file)

    if len(comparison_rows) != 2:
        raise ValueError(f"Expected 2 comparison rows, found {len(comparison_rows)}.")
    comparison_models = {row["model"].strip() for row in comparison_rows}
    if comparison_models != {"Original", "Augmented"}:
        raise ValueError(
            "Expected comparison rows for exactly Original and Augmented models."
        )
    if any(row["split"].strip() != "validation" for row in comparison_rows):
        raise ValueError('Expected both comparison rows to have split == "validation".')
    if len(baseline_rows) < 7 or [row["fold"] for row in baseline_rows[:5]] != [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]:
        raise ValueError("Expected five baseline fold rows before mean/std rows.")
    if len(weight_decay_rows) < 7 or [
        row["fold"] for row in weight_decay_rows[:5]
    ] != ["1", "2", "3", "4", "5"]:
        raise ValueError("Expected five weight-decay fold rows before mean/std rows.")
    if not isinstance(weight_decay_config, dict):
        raise ValueError("Weight-decay experiment config must be a JSON object.")
    missing_hyperparameters = [
        key for key in CONFIG_HYPERPARAMETERS if key not in weight_decay_config
    ]
    if missing_hyperparameters:
        raise ValueError(
            "Weight-decay experiment config is missing hyperparameter(s): "
            + ", ".join(missing_hyperparameters)
        )
    validate_numeric_metrics(comparison_rows, "comparison CSV")
    validate_numeric_metrics(baseline_rows[:5], "baseline CSV")
    validate_numeric_metrics(weight_decay_rows[:5], "weight-decay CSV")

    return (
        comparison_rows,
        baseline_rows[:5],
        weight_decay_rows[:5],
        weight_decay_config,
    )


def metric_value(row: dict[str, str], source_name: str, metric_name: str) -> float | None:
    value = row.get(source_name, "").strip()
    if not value:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {source_name} is not finite: {value!r}")
    return result


def metrics_from_row(row: dict[str, str]) -> dict[str, float]:
    metric_sources = {
        "top1_accuracy": "top1_accuracy",
        "top5_accuracy": "top5_accuracy",
        "validation_loss": "validation_loss",
        "best_validation_loss": "best_validation_loss",
        "epochs_trained": "epochs_trained",
        "training_loss": "training_loss_at_best_validation_loss",
        "training_accuracy": "training_accuracy_at_best_validation_loss",
    }
    metrics: dict[str, float] = {}
    for metric_name, source_name in metric_sources.items():
        value = metric_value(row, source_name, metric_name)
        if value is not None:
            metrics[metric_name] = value
    return metrics


def params_from_config(config: dict[str, Any]) -> dict[str, str]:
    return {key: str(config[key]) for key in CONFIG_HYPERPARAMETERS}


def existing_run_id(client: MlflowClient, experiment_id: str, historical_id: str) -> str | None:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.historical_run_id = '{historical_id}'",
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def log_run(
    client: MlflowClient,
    experiment_id: str,
    *,
    name: str,
    historical_id: str,
    tags: dict[str, str],
    metrics: dict[str, float],
    params: dict[str, str],
    artifacts: tuple[Path, ...],
) -> str:
    run_id = existing_run_id(client, experiment_id, historical_id)
    if run_id is not None:
        print(f"{name}: {run_id} (already exists)")
        return run_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=name,
        tags={**COMMON_TAGS, **tags, "historical_run_id": historical_id},
    ) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        run_id = run.info.run_id

    print(f"{name}: {run_id}")
    return run_id


def main() -> None:
    # Preflight all files and rows before creating or searching MLflow runs.
    require_files()
    comparison_rows, baseline_rows, weight_decay_rows, weight_decay_config = load_inputs()

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id
    client = MlflowClient()

    logged_run_ids: list[str] = []

    for row in comparison_rows:
        model = row["model"].strip()
        historical_id = f"validation-comparison-{model.lower()}"
        logged_run_ids.append(
            log_run(
                client,
                experiment_id,
                name=f"Validation comparison - {model}",
                historical_id=historical_id,
                tags={"experiment_type": "validation_comparison", "model_variant": model},
                metrics=metrics_from_row(row),
                params={},
                artifacts=(COMPARISON_FILE,),
            )
        )

    for row in baseline_rows:
        fold = row["fold"].strip()
        logged_run_ids.append(
            log_run(
                client,
                experiment_id,
                name=f"Baseline Transformer - fold {fold}",
                historical_id=f"baseline-transformer-fold-{fold}",
                tags={"experiment_type": "cross_validation", "fold": fold},
                metrics=metrics_from_row(row),
                params={},
                artifacts=(BASELINE_FILE,),
            )
        )

    for row in weight_decay_rows:
        fold = row["fold"].strip()
        logged_run_ids.append(
            log_run(
                client,
                experiment_id,
                name=f"Weight decay 0.01 Transformer - fold {fold}",
                historical_id=f"weight-decay-0.01-transformer-fold-{fold}",
                tags={"experiment_type": "cross_validation_weight_decay", "fold": fold},
                metrics=metrics_from_row(row),
                params=params_from_config(weight_decay_config),
                artifacts=(WEIGHT_DECAY_FILE, WEIGHT_DECAY_CONFIG_FILE),
            )
        )

    print(f"Total successfully logged runs: {len(logged_run_ids)}")


if __name__ == "__main__":
    main()
