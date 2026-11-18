"""In-memory lighting measurements for OpenCV BGR frames."""

from __future__ import annotations

import cv2
import numpy as np


def analyze_lighting(frame_bgr: np.ndarray) -> dict[str, float]:
    """Return brightness statistics for one uint8 OpenCV BGR frame.

    The frame is converted to an 8-bit grayscale image and is not saved,
    displayed, or transmitted. Percentage values are expressed from 0 to 100.
    """
    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr must be a NumPy array.")
    if frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must have dtype uint8.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3).")
    if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
        raise ValueError("frame_bgr must not be empty.")

    grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    brightness = grayscale.astype(np.float32, copy=False)
    pixel_count = brightness.size

    return {
        "mean_brightness": float(np.mean(brightness)),
        "median_brightness": float(np.median(brightness)),
        "std_brightness": float(np.std(brightness)),
        "percent_below_50": float(np.count_nonzero(brightness < 50) / pixel_count * 100),
        "percent_above_200": float(np.count_nonzero(brightness > 200) / pixel_count * 100),
    }
