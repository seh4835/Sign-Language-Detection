"""Evaluate the selected temporal baseline once on the corrected held-out test view."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_yolo_pose_hands_lstm_v2 import FeatureNormalizer, collect_sequences  # noqa: E402

AUDIT = ROOT / "models" / "yolo_pose_hands_lstm_v2" / "final_test_evaluation" / "pre_evaluation_audit"
OUT = AUDIT.parent
BASELINE = ROOT / "models" / "yolo_pose_hands_lstm_v2" / "baseline_diagnostics"
MAPPING_PATH = ROOT / "data" / "metadata" / "label_mapping.json"
EXPECTED_TEST_ROWS = 248
LABEL_COUNT = 100


def summary_features(sequences: np.ndarray) -> np.ndarray:
    return np.concatenate((sequences.mean(axis=1), sequences.std(axis=1)), axis=1)


def load_manifest(name: str, mapping: dict[str, int]) -> pd.DataFrame:
    rows = pd.read_csv(AUDIT / name, dtype={"video_id": str, "signer_id": str})
    required = {"dataset", "video_id", "signer_id", "gloss", "feature_path_abs", "video_path_abs"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if rows.video_id.duplicated().any():
        raise ValueError(f"{name} contains duplicate video IDs.")
    if not rows.feature_path_abs.map(lambda path: Path(path).is_file()).all():
        raise FileNotFoundError(f"{name} contains missing feature paths.")
    if not rows.video_path_abs.map(lambda path: Path(path).is_file()).all():
        raise FileNotFoundError(f"{name} contains missing video paths.")
    labels = rows.gloss.astype(str).map(mapping)
    if labels.isna().any() or not labels.between(0, LABEL_COUNT - 1).all():
        raise ValueError(f"{name} contains labels absent from the canonical mapping.")
    rows["feature_path"] = rows.feature_path_abs.map(Path)
    rows["label"] = labels.astype(int)
    return rows


def verify_isolation(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    if len(train) != 1181 or len(validation) != 232 or len(test) != EXPECTED_TEST_ROWS:
        raise ValueError("Corrected derived manifest row counts do not match the approved audit.")
    development = pd.concat([train, validation], ignore_index=True)
    if set(development.video_id) & set(test.video_id):
        raise ValueError("Video ID overlap remains between development and test.")
    development_signers = set(zip(development.dataset.astype(str), development.signer_id.astype(str)))
    test_signers = set(zip(test.dataset.astype(str), test.signer_id.astype(str)))
    if development_signers & test_signers:
        raise ValueError("Signer overlap remains between development and test.")
    audit = json.loads((AUDIT / "integrity_report.json").read_text(encoding="utf-8"))
    if audit["test_rows_preserved"] != EXPECTED_TEST_ROWS:
        raise ValueError("Integrity report does not preserve all approved test rows.")
    if audit["sha256_overlap_development_test"]:
        raise ValueError("Integrity report records development/test content overlap.")
    if audit["signer_overlap_development_test"]:
        raise ValueError("Integrity report records development/test signer overlap.")


def main() -> None:
    outputs = [
        OUT / "test_metrics.json",
        OUT / "test_predictions.csv",
        OUT / "test_confusion_matrix.csv",
        OUT / "test_per_class_metrics.csv",
        OUT / "final_test_report.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing evaluation outputs: {existing}")

    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    train = load_manifest("train_derived_manifest.csv", mapping)
    validation = load_manifest("validation_derived_manifest.csv", mapping)
    test = load_manifest("test_derived_manifest.csv", mapping)
    verify_isolation(train, validation, test)

    # The only arrays passed to fit are derived from the approved train manifest.
    train_features, train_presence = collect_sequences(train)
    normalizer = FeatureNormalizer.fit(train_features, train_presence)
    train_sequences = normalizer.transform_lstm(train_features, train_presence)
    train_summary = summary_features(train_sequences)
    train_labels = train.label.to_numpy()

    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    model.fit(train_summary, train_labels)

    # Test data is loaded only after fitting and is used exactly once for prediction.
    test_features, test_presence = collect_sequences(test)
    test_sequences = normalizer.transform_lstm(test_features, test_presence)
    test_predictions = model.predict(summary_features(test_sequences))
    test_labels = test.label.to_numpy()
    if len(test_predictions) != EXPECTED_TEST_ROWS:
        raise RuntimeError("The number of test predictions is not 248.")

    labels = list(range(LABEL_COUNT))
    glosses = [gloss for gloss, _ in sorted(mapping.items(), key=lambda item: int(item[1]))]
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(test_labels, test_predictions, labels=labels)
    prediction_counts = np.bincount(test_predictions, minlength=LABEL_COUNT)
    correct_counts = np.diag(matrix)
    report = classification_report(
        test_labels, test_predictions, labels=labels, target_names=glosses, zero_division=0, output_dict=True
    )

    class_metrics = pd.DataFrame(
        {
            "label": labels,
            "gloss": glosses,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_count": prediction_counts,
            "correct_count": correct_counts,
        }
    )
    class_metrics.to_csv(OUT / "test_per_class_metrics.csv", index=False)
    confusion = pd.DataFrame(matrix, index=glosses, columns=glosses)
    confusion.index.name = "true_gloss"
    confusion.to_csv(OUT / "test_confusion_matrix.csv")

    predictions = test[["original_split", "dataset", "video_id", "signer_id", "gloss", "label"]].copy()
    predictions = predictions.rename(columns={"label": "true_label"})
    predictions["predicted_label"] = test_predictions
    predictions["predicted_gloss"] = [glosses[label] for label in test_predictions]
    predictions["correct"] = test_labels == test_predictions
    predictions.to_csv(OUT / "test_predictions.csv", index=False)

    distribution = pd.DataFrame({"label": labels, "gloss": glosses, "predicted_count": prediction_counts})
    top_predictions = distribution[distribution.predicted_count > 0].sort_values(
        ["predicted_count", "gloss"], ascending=[False, True]
    )
    validation_summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))["validation"]
    metrics = {
        "algorithm": {
            "classifier": "sklearn.linear_model.LogisticRegression",
            "max_iter": 500,
            "solver": "lbfgs",
            "random_state": 42,
            "features": "concatenate(sequence.mean(axis=1), sequence.std(axis=1))",
            "input_dim": int(train_summary.shape[1]),
            "normalization": "FeatureNormalizer fitted on train_derived_manifest.csv only",
        },
        "data": {
            "train_samples_fitted": int(len(train)),
            "validation_samples_used_for_fit": 0,
            "test_samples_evaluated": int(len(test)),
            "test_predictions": int(len(test_predictions)),
            "test_video_ids_unique": int(test.video_id.nunique()),
            "evaluation_passes": 1,
        },
        "test": {
            "accuracy": float(accuracy_score(test_labels, test_predictions)),
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "predicted_class_count": int(np.count_nonzero(prediction_counts)),
            "correctly_recognized_class_count": int(np.count_nonzero(correct_counts)),
            "top_predictions": [
                {"gloss": row.gloss, "count": int(row.predicted_count)}
                for row in top_predictions.head(20).itertuples()
            ],
        },
        "validation_comparison": validation_summary,
        "outputs": {
            "predictions": "test_predictions.csv",
            "confusion_matrix": "test_confusion_matrix.csv",
            "per_class_metrics": "test_per_class_metrics.csv",
        },
    }
    (OUT / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Final Held-Out Test Evaluation",
        "",
        "The selected temporal mean/variance baseline was fit once using only the corrected derived training view and evaluated once on the untouched corrected test view.",
        "",
        "## Protocol",
        "",
        f"- Train samples used for fitting: **{len(train)}**",
        "- Validation samples used for fitting: **0**",
        f"- Test samples evaluated exactly once: **{len(test)}**",
        "- Algorithm: `LogisticRegression(max_iter=500, solver=\"lbfgs\", random_state=42)`",
        "- Features: sequence mean concatenated with sequence standard deviation (360 dimensions)",
        "- Normalization: `FeatureNormalizer` fitted on training data only",
        "",
        "## Validation Comparison",
        "",
        "| Metric | Validation | Final test |",
        "|---|---:|---:|",
        f"| Accuracy | {validation_summary['accuracy']:.6f} | {metrics['test']['accuracy']:.6f} |",
        f"| Macro precision | {validation_summary['macro_precision']:.6f} | {metrics['test']['macro_precision']:.6f} |",
        f"| Macro recall | {validation_summary['macro_recall']:.6f} | {metrics['test']['macro_recall']:.6f} |",
        f"| Macro-F1 | {validation_summary['macro_f1']:.6f} | {metrics['test']['macro_f1']:.6f} |",
        f"| Weighted-F1 | not saved | {metrics['test']['weighted_f1']:.6f} |",
        "",
        "## Final Test Results",
        "",
        f"- Accuracy: **{metrics['test']['accuracy']:.6f}**",
        f"- Macro precision: **{metrics['test']['macro_precision']:.6f}**",
        f"- Macro recall: **{metrics['test']['macro_recall']:.6f}**",
        f"- Macro-F1: **{metrics['test']['macro_f1']:.6f}**",
        f"- Weighted-F1: **{metrics['test']['weighted_f1']:.6f}**",
        f"- Predicted classes: **{metrics['test']['predicted_class_count']}/100**",
        f"- Correctly recognized classes: **{metrics['test']['correctly_recognized_class_count']}/100**",
        "",
        "Per-class metrics are in `test_per_class_metrics.csv`; the complete confusion matrix is in `test_confusion_matrix.csv`.",
    ]
    (OUT / "final_test_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
