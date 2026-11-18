"""Fit and save the selected temporal baseline for webcam inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_yolo_pose_hands_lstm_v2 import collect_sequences  # noqa: E402
from src.models.temporal_baseline import save_artifact, summary_features  # noqa: E402
from src.models.yolo_pose_hands_lstm_v2 import FeatureNormalizer  # noqa: E402


AUDIT = ROOT / "models" / "yolo_pose_hands_lstm_v2" / "final_test_evaluation" / "pre_evaluation_audit"
OUT = ROOT / "models" / "yolo_pose_hands_lstm_v2" / "webcam_baseline_artifacts"
MAPPING_PATH = ROOT / "data" / "metadata" / "label_mapping.json"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {OUT}")
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    train = __import__("pandas").read_csv(
        AUDIT / "train_derived_manifest.csv", dtype={"video_id": str, "signer_id": str}
    )
    if len(train) != 1181:
        raise ValueError("Expected exactly 1,181 corrected training rows.")
    train["feature_path"] = train.feature_path_abs.map(Path)
    train_labels = train.gloss.astype(str).map(mapping).to_numpy()
    features, presence = collect_sequences(train)
    normalizer = FeatureNormalizer.fit(features, presence)
    normalized = normalizer.transform_lstm(features, presence)
    classifier = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
    classifier.fit(summary_features(normalized), train_labels)
    save_artifact(OUT, classifier, normalizer, mapping)
    config = {
        "classifier": "sklearn.linear_model.LogisticRegression",
        "max_iter": 500,
        "solver": "lbfgs",
        "random_state": 42,
        "features": "concatenate(sequence.mean(axis=1), sequence.std(axis=1))",
        "input_dim": 360,
        "normalization": "FeatureNormalizer fitted on corrected train_derived_manifest.csv only",
        "training_samples": len(train),
        "validation_samples_used": 0,
        "test_samples_used": 0,
    }
    (OUT / "algorithm_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
