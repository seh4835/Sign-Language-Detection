"""Pydantic schemas for the sign prediction API."""

from __future__ import annotations

import math

from pydantic import BaseModel, field_validator


FRAME_COUNT = 30
FEATURE_COUNT = 180


class PredictionRequest(BaseModel):
    """A single preprocessed sign-language feature sequence."""

    sequence: list[list[float]]

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != FRAME_COUNT:
            raise ValueError(f"sequence must contain exactly {FRAME_COUNT} frames")
        for frame_index, frame in enumerate(value):
            if len(frame) != FEATURE_COUNT:
                raise ValueError(
                    f"sequence frame {frame_index} must contain exactly "
                    f"{FEATURE_COUNT} features"
                )
            if not all(math.isfinite(feature) for feature in frame):
                raise ValueError(
                    f"sequence frame {frame_index} contains non-finite values"
                )
        return value


class PredictionResponse(BaseModel):
    """Prediction result returned by the existing temporal baseline."""

    predicted_sign: str
    confidence: float | None
    status: str
