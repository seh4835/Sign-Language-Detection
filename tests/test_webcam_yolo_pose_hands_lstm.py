import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import webcam_yolo_pose_hands_lstm as app


class FakeCamera:
    def __init__(self, *_args):
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.read_count += 1
        return True, np.zeros((64, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeExtractor:
    def __init__(self):
        self.last_presence = {"body_pose": False, "left_hand": False, "right_hand": False}
        self.closed = False

    def process_frame(self, _frame):
        return np.zeros(177, dtype=np.float32)

    def draw_annotations(self, frame):
        return frame

    def close(self):
        self.closed = True


class WebcamSmokeTest(unittest.TestCase):
    def test_uniform_sampling_short_and_long_intervals(self):
        short = np.arange(30 * 180, dtype=np.float32).reshape(30, 180)
        np.testing.assert_array_equal(app.sample_uniform_sequence(short), short)
        long = np.arange(60 * 180, dtype=np.float32).reshape(60, 180)
        sampled = app.sample_uniform_sequence(long)
        self.assertEqual(sampled.shape, (30, 180))
        np.testing.assert_array_equal(sampled[:, 0], np.linspace(0, 59, 30, dtype=np.int64) * 180)
        with self.assertRaises(ValueError):
            app.sample_uniform_sequence(short[:29])

    def test_capture_prediction_and_shutdown_path(self):
        camera, extractor = FakeCamera(), FakeExtractor()
        keys = {"count": 0}

        def wait_key(_delay):
            keys["count"] += 1
            return ord("q") if keys["count"] >= 30 else -1

        with patch.object(sys, "argv", ["webcam_yolo_pose_hands_lstm.py"]), \
             patch.object(app.cv2, "VideoCapture", return_value=camera), \
             patch.object(app.YoloPoseHandsExtractor, "from_checkpoint", return_value=extractor), \
             patch.object(app.cv2, "imshow"), patch.object(app.cv2, "waitKey", side_effect=wait_key), \
             patch.object(app.cv2, "destroyAllWindows"):
            app.main()
        self.assertEqual(camera.read_count, 30)
        self.assertTrue(camera.released)
        self.assertTrue(extractor.closed)
        self.assertEqual(keys["count"], 30)


if __name__ == "__main__":
    unittest.main()
