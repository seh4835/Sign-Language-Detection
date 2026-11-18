import numpy as np

from src.keypoints.mediapipe_hands import MediaPipeHandsExtractor


class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class HandLandmarks:
    def __init__(self, offset):
        self.landmark = [
            Landmark(offset + index, offset + index + 0.1, offset + index + 0.2)
            for index in range(21)
        ]


def test_missing_hand_uses_existing_zero_fill_shape():
    missing = MediaPipeHandsExtractor._extract_hand(None)
    assert missing.shape == (21, 3)
    assert missing.dtype == np.float32
    assert np.count_nonzero(missing) == 0


def test_hand_extraction_preserves_landmark_order():
    hand = MediaPipeHandsExtractor._extract_hand(HandLandmarks(1.0))
    assert hand.shape == (21, 3)
    np.testing.assert_allclose(hand[0], [1.0, 1.1, 1.2])
    np.testing.assert_allclose(hand[-1], [21.0, 21.1, 21.2])


def test_process_frame_preserves_left_then_right_order_without_model():
    extractor = object.__new__(MediaPipeHandsExtractor)
    extractor.last_results = type(
        "Results",
        (),
        {
            "multi_hand_landmarks": [
                HandLandmarks(10.0),
                HandLandmarks(20.0),
            ],
            "multi_handedness": [
                type(
                    "Handedness",
                    (),
                    {"classification": [type("Class", (), {"label": "Right"})()]},
                )(),
                type(
                    "Handedness",
                    (),
                    {"classification": [type("Class", (), {"label": "Left"})()]},
                )(),
            ],
        },
    )()
    extractor.hands = type(
        "Hands",
        (),
        {"process": lambda _, frame: extractor.last_results},
    )()

    left, right, combined = extractor.process_frame(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )

    np.testing.assert_allclose(left[0], [20.0, 20.1, 20.2])
    np.testing.assert_allclose(right[0], [10.0, 10.1, 10.2])
    assert combined.shape == (126,)
    np.testing.assert_allclose(combined[:63], left.reshape(-1))
    np.testing.assert_allclose(combined[63:], right.reshape(-1))
