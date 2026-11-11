"""FastAPI service for the existing sign-language temporal baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.linear_model import LogisticRegression

from src.models.temporal_baseline import load_artifact, predict_sequence

from .schemas import PredictionRequest, PredictionResponse


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    ROOT
    / "models"
    / "yolo_pose_hands_lstm_v2"
    / "webcam_baseline_artifacts"
)

app = FastAPI(title="Sign Language Recognition API")

_artifact_cache: tuple[object, object, list[str]] | None = None
_artifact_error: str | None = None


def _load_artifacts() -> tuple[object, object, list[str]] | None:
    global _artifact_cache, _artifact_error
    if _artifact_cache is not None:
        return _artifact_cache
    if _artifact_error is not None:
        return None
    try:
        _artifact_cache = load_artifact(ARTIFACTS)
        classifier, normalizer, labels = _artifact_cache
        if isinstance(classifier, LogisticRegression) and not hasattr(
            classifier, "multi_class"
        ):
            classifier.multi_class = "auto"
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        _artifact_error = str(error)
        return None
    return _artifact_cache


def _reset_artifact_cache() -> None:
    """Reset cached loading state for isolated API tests."""
    global _artifact_cache, _artifact_error
    _artifact_cache = None
    _artifact_error = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Report whether the existing model artifacts are available."""
    artifacts = _load_artifacts()
    if artifacts is None:
        return {
            "status": "unavailable",
            "model_loaded": False,
            "detail": _artifact_error or "Model artifacts are unavailable.",
        }
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict a sign from one 30-frame, 180-feature sequence."""
    artifacts = _load_artifacts()
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail=_artifact_error or "Model artifacts are unavailable.",
        )

    classifier, normalizer, labels = artifacts
    sequence = np.asarray(request.sequence, dtype=np.float32)
    predicted_label, confidence = predict_sequence(
        classifier,
        normalizer,
        sequence,
    )
    if not 0 <= predicted_label < len(labels):
        raise HTTPException(
            status_code=500,
            detail="Model returned a label outside the artifact label mapping.",
        )
    return PredictionResponse(
        predicted_sign=labels[predicted_label],
        confidence=confidence,
        status="ok",
    )
