import numpy as np
import pytest

from src.keypoints.lighting_analysis import analyze_lighting


def test_analyze_lighting_returns_statistics_for_synthetic_bgr_image():
    frame = np.array(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[100, 100, 100], [220, 220, 220]],
        ],
        dtype=np.uint8,
    )

    measurements = analyze_lighting(frame)

    assert measurements["mean_brightness"] == pytest.approx(143.75)
    assert measurements["median_brightness"] == pytest.approx(160.0)
    assert measurements["std_brightness"] == pytest.approx(100.9564)
    assert measurements["percent_below_50"] == pytest.approx(25.0)
    assert measurements["percent_above_200"] == pytest.approx(50.0)


def test_analyze_lighting_converts_color_bgr_to_luminance():
    frame = np.array([[[255, 0, 0]]], dtype=np.uint8)

    measurements = analyze_lighting(frame)

    assert measurements["mean_brightness"] == pytest.approx(29.0)
    assert measurements["median_brightness"] == pytest.approx(29.0)


@pytest.mark.parametrize(
    "frame",
    [
        None,
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 1), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_analyze_lighting_rejects_invalid_frames(frame):
    with pytest.raises(ValueError):
        analyze_lighting(frame)
