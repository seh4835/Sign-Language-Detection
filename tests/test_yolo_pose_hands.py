import numpy as np

from src.keypoints.yolo_pose_hands import YoloPoseHandsExtractor, preprocess_features


def test_feature_layout_and_order():
    body = np.arange(51, dtype=np.float32).reshape(17, 3)
    left = np.arange(63, dtype=np.float32).reshape(21, 3) + 100
    right = np.arange(63, dtype=np.float32).reshape(21, 3) + 200
    features = YoloPoseHandsExtractor.combine_features(body, left, right)
    assert features.shape == (177,)
    np.testing.assert_array_equal(features[:51], body.reshape(-1))
    np.testing.assert_array_equal(features[51:114], left.reshape(-1))
    np.testing.assert_array_equal(features[114:], right.reshape(-1))


def test_missing_detections_zero_fill_without_presence_confusion():
    body, body_present = YoloPoseHandsExtractor._extract_body(
        type("Result", (), {"keypoints": None})()
    )
    hand, hand_present = YoloPoseHandsExtractor._extract_hand(None)
    assert body.shape == (17, 3)
    assert hand.shape == (21, 3)
    assert not body_present
    assert not hand_present
    assert np.count_nonzero(body) == 0
    assert np.count_nonzero(hand) == 0


def test_single_frame_and_sequence_shapes():
    feature = YoloPoseHandsExtractor.combine_features(
        np.zeros((17, 3), dtype=np.float32),
        np.zeros((21, 3), dtype=np.float32),
        np.zeros((21, 3), dtype=np.float32),
    )
    sequence = np.stack([feature] * 30)
    assert feature.shape == (177,)
    assert sequence.shape == (30, 177)
    assert np.isfinite(sequence).all()


def test_preprocessing_normalizes_body_pixels_and_preserves_hand_conventions():
    raw = np.zeros(177, dtype=np.float32)
    raw[0:3] = [640.0, 360.0, 0.8]
    raw[51:54] = [0.25, 0.5, -2.0]
    processed, indicators = preprocess_features(
        raw,
        frame_width=1280,
        frame_height=720,
        presence={"body_pose": True, "left_hand": True, "right_hand": False},
    )
    np.testing.assert_allclose(processed[:3], [0.5, 0.5, 0.8])
    np.testing.assert_allclose(processed[51:54], [0.25, 0.5, -1.0])
    assert indicators.tolist() == [1.0, 1.0, 0.0]
    assert np.count_nonzero(processed[114:]) == 0


def test_preprocessing_rejects_missing_or_invalid_presence():
    raw = np.zeros(177, dtype=np.float32)
    try:
        preprocess_features(raw, 640, 480, {"body_pose": True})
    except ValueError as error:
        assert "presence" in str(error).lower()
    else:
        raise AssertionError("Expected incomplete presence indicators to fail.")
