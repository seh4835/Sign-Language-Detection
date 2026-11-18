"""Run reproducible Optuna tuning on the development-only Transformer folds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cross_validate_transformer import set_seed, train_fold
from src.cross_validation import (
    N_FOLDS,
    RANDOM_STATE,
    create_fold_assignments,
    load_development_data,
)
from src.models.transformer_pose_hands import load_label_mapping


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
OUTPUT = ROOT / "reports" / "optuna" / "best_trial.json"
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Sign Language Recognition - Optuna"
MAX_EPOCHS = 40
PATIENCE = 8
N_TRIALS = 3


def build_development_assignments():
    development = load_development_data(
        META / "train_signer_disjoint.csv",
        META / "validation_signer_disjoint.csv",
    )
    assignments, _ = create_fold_assignments(
        development,
        n_splits=N_FOLDS,
        random_state=RANDOM_STATE,
    )
    return assignments


def objective(
    trial: optuna.Trial,
    assignments,
    mapping: dict[str, int],
) -> float:
    learning_rate = trial.suggest_float(
        "learning_rate", 0.0001, 0.001, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 0.00001, 0.01, log=True
    )
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    fold_results = []
    for fold in range(1, N_FOLDS + 1):
        set_seed(RANDOM_STATE + fold)
        result, _, _ = train_fold(
            fold,
            assignments,
            mapping,
            epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch_size=int(batch_size),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            save_checkpoint=False,
        )
        fold_results.append(result)

    mean_validation_loss = float(
        np.mean([float(result["best_validation_loss"]) for result in fold_results])
    )
    mean_top1_accuracy = float(
        np.mean([float(result["top1_accuracy"]) for result in fold_results])
    )
    mean_top5_accuracy = float(
        np.mean([float(result["top5_accuracy"]) for result in fold_results])
    )

    mlflow_params = {
        "trial_number": str(trial.number),
        "learning_rate": str(learning_rate),
        "weight_decay": str(weight_decay),
        "dropout": str(dropout),
        "batch_size": str(batch_size),
        "epochs": str(MAX_EPOCHS),
        "patience": str(PATIENCE),
        "folds": str(N_FOLDS),
    }
    with mlflow.start_run(
        run_name=f"Optuna trial {trial.number}",
        tags={
            "architecture": "Transformer",
            "experiment_type": "optuna_tuning",
            "run_origin": "optuna_tuning",
            "trial_number": str(trial.number),
        },
    ):
        mlflow.log_params(mlflow_params)
        mlflow.log_metrics(
            {
                "mean_validation_loss": mean_validation_loss,
                "mean_top1_accuracy": mean_top1_accuracy,
                "mean_top5_accuracy": mean_top5_accuracy,
            }
        )

    print(
        f"Trial {trial.number}: params={trial.params} | "
        f"mean_validation_loss={mean_validation_loss:.6f} | "
        f"mean_top1_accuracy={mean_top1_accuracy:.6f} | "
        f"mean_top5_accuracy={mean_top5_accuracy:.6f}"
    )
    trial.set_user_attr("mean_top1_accuracy", mean_top1_accuracy)
    trial.set_user_attr("mean_top5_accuracy", mean_top5_accuracy)
    return mean_validation_loss


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Optuna report: {OUTPUT}"
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mapping = load_label_mapping(META / "label_mapping.json")
    assignments = build_development_assignments()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(
        lambda trial: objective(trial, assignments, mapping),
        n_trials=N_TRIALS,
    )

    best_trial = study.best_trial
    best_metrics = {
        "mean_validation_loss": best_trial.value,
        "mean_top1_accuracy": best_trial.user_attrs["mean_top1_accuracy"],
        "mean_top5_accuracy": best_trial.user_attrs["mean_top5_accuracy"],
    }
    output = {
        "trial_number": best_trial.number,
        "parameters": best_trial.params,
        "validation_metrics": best_metrics,
        "folds": N_FOLDS,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "random_seed": RANDOM_STATE,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(
        f"Best trial {best_trial.number}: params={best_trial.params} | "
        f"mean_validation_loss={best_metrics['mean_validation_loss']:.6f} | "
        f"mean_top1_accuracy={best_metrics['mean_top1_accuracy']:.6f} | "
        f"mean_top5_accuracy={best_metrics['mean_top5_accuracy']:.6f}"
    )
    print(f"Best trial report saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
