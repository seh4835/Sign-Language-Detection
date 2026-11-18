import numpy as np
import pytest

from src.api.webcam_capture import (
    FRAME_COUNT,
    IncompleteCaptureError,
    WebcamCapture,
    sample_uniform_frames,
)


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


class FakeExtractor:
    def __init__(self):
        self.index = 0
        self.last_presence = {
            "body_pose": True,
            "left_hand": self.index % 2 == 0,
            "right_hand": self.index % 3 == 0,
        }

    def process_frame(self, frame):
        value = float(frame[0, 0, 0])
        self.last_presence = {
            "body_pose": True,
            "left_hand": value % 2 == 0,
            "right_hand": value % 3 == 0,
        }
        self.index += 1
        return np.full(177, value, dtype=np.float32)

    def draw_annotations(self, frame):
        return frame

    def close(self):
        pass


def make_session(count=FRAME_COUNT):
    frames = [
        np.full((2, 4, 3), index, dtype=np.uint8)
        for index in range(count)
    ]
    return WebcamCapture(FakeExtractor(), FakeCapture(frames))


def test_capture_collects_exactly_30_chronological_frames_and_shapes():
    result = make_session().capture_one_sign(display=False)
    assert result.raw_features.shape == (30, 177)
    assert result.presence.shape == (30, 3)
    assert result.lstm_features.shape == (30, 180)
    np.testing.assert_array_equal(result.raw_features[:, 0], np.arange(30))
    np.testing.assert_array_equal(result.lstm_features[:, 177:], result.presence)


def test_presence_indicators_are_binary_and_ordered():
    result = make_session().capture_one_sign(display=False)
    assert set(np.unique(result.presence)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(result.presence[0], [1, 1, 1])
    np.testing.assert_array_equal(result.presence[1], [1, 0, 0])


def test_incomplete_capture_is_rejected():
    session = make_session(count=29)
    with pytest.raises(IncompleteCaptureError, match="29/30"):
        session.capture_one_sign(display=False)


def test_close_releases_camera():
    session = make_session()
    capture = session.capture
    session.close()
    assert capture.released


def test_uniform_interval_sampling_includes_endpoints():
    frames = [np.asarray([index]) for index in range(60)]
    sampled = sample_uniform_frames(frames)
    np.testing.assert_array_equal(
        np.asarray(sampled).ravel(), np.linspace(0, 59, 30, dtype=np.int64)
    )


def test_uniform_interval_sampling_rejects_short_interval():
    with pytest.raises(IncompleteCaptureError, match="29/30"):
        sample_uniform_frames([np.asarray([index]) for index in range(29)])
