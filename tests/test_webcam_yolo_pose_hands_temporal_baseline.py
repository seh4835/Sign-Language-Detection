import csv
import sys
from unittest.mock import patch

import numpy as np
import pytest

import webcam_yolo_pose_hands_temporal_baseline as app
from scripts.log_lighting_experiment import append_experiment_record


def test_aggregate_lighting_statistics_uses_means():
    statistics = [
        {
            "mean_brightness": 10,
            "median_brightness": 8,
            "std_brightness": 2,
            "percent_below_50": 90,
            "percent_above_200": 1,
        },
        {
            "mean_brightness": 30,
            "median_brightness": 28,
            "std_brightness": 6,
            "percent_below_50": 70,
            "percent_above_200": 5,
        },
    ]

    result = app.aggregate_lighting_statistics(statistics)

    assert result == {
        "mean_brightness": 20.0,
        "median_brightness": 18.0,
        "std_brightness": 4.0,
        "percent_below_50": 80.0,
        "percent_above_200": 3.0,
    }


def test_aggregate_lighting_statistics_ignores_unavailable_measurements():
    result = app.aggregate_lighting_statistics([None, {}, None])

    assert result is None


def test_normal_mode_aggregate_tolerates_missing_measurements():
    valid = {
        "mean_brightness": 20,
        "median_brightness": 18,
        "std_brightness": 4,
        "percent_below_50": 70,
        "percent_above_200": 5,
    }

    assert app.aggregate_lighting_statistics([valid, None]) == {
        key: float(value) for key, value in valid.items()
    }


def test_experiment_interval_with_missing_frame_has_no_complete_aggregate():
    valid = {
        "mean_brightness": 20,
        "median_brightness": 18,
        "std_brightness": 4,
        "percent_below_50": 70,
        "percent_above_200": 5,
    }

    aggregate, missing_count = app.aggregate_complete_lighting_interval(
        [valid, None, valid], expected_count=3
    )

    assert aggregate is None
    assert missing_count == 1


def test_experiment_interval_rejects_invalid_statistics():
    valid = {
        "mean_brightness": 20,
        "median_brightness": 18,
        "std_brightness": 4,
        "percent_below_50": 70,
        "percent_above_200": 5,
    }
    invalid = {**valid, "std_brightness": float("nan")}

    aggregate, missing_count = app.aggregate_complete_lighting_interval(
        [valid, invalid], expected_count=2
    )

    assert aggregate is None
    assert missing_count == 1


def test_build_lighting_experiment_record_has_expected_scalar_fields():
    statistics = {
        "mean_brightness": 101.5,
        "median_brightness": 99.0,
        "std_brightness": 24.25,
        "percent_below_50": 8.0,
        "percent_above_200": 2.5,
    }

    record = app.build_lighting_experiment_record(
        experiment_id="EXP01",
        participant_id="P01",
        lighting_condition="dim",
        predicted_sign="hello",
        actual_sign="hello",
        lighting_statistics=statistics,
    )

    assert record == {
        "experiment_id": "EXP01",
        "participant_id": "P01",
        "lighting_condition": "dim",
        **statistics,
        "predicted_sign": "hello",
        "actual_sign": "hello",
    }
    assert all(not isinstance(value, np.ndarray) for value in record.values())


def test_logging_failure_reports_error_without_mutating_prediction(monkeypatch):
    prediction = ("hello", 0.875)

    def fail_logging(_record):
        raise OSError("synthetic disk error")

    monkeypatch.setattr(app, "append_experiment_record", fail_logging)
    error = app.try_log_experiment_record({"synthetic": "record"})

    assert error == (
        "Experiment CSV logging failed; prediction remains available: "
        "synthetic disk error"
    )
    assert prediction == ("hello", 0.875)


def test_aggregate_lighting_statistics_uses_every_interval_measurement():
    statistics = [
        {
            "mean_brightness": value,
            "median_brightness": value,
            "std_brightness": value,
            "percent_below_50": value,
            "percent_above_200": value,
        }
        for value in (10, 20, 30, 40)
    ]

    result = app.aggregate_lighting_statistics(statistics)

    assert result == {
        "mean_brightness": 25.0,
        "median_brightness": 25.0,
        "std_brightness": 25.0,
        "percent_below_50": 25.0,
        "percent_above_200": 25.0,
    }


class FakeCamera:
    def __init__(self, opened=True, reads=()):
        self.opened = opened
        self.reads = list(reads)
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        return self.reads.pop(0) if self.reads else (False, None)

    def release(self):
        self.released = True


def test_open_camera_tries_indices_until_one_opens_and_returns_a_frame():
    frame = np.ones((8, 10, 3), dtype=np.uint8)
    failed_open = FakeCamera(opened=False)
    failed_read = FakeCamera(reads=[(False, None)])
    working = FakeCamera(reads=[(True, frame)])
    cameras = {0: failed_open, 1: failed_read, 2: working}

    with patch.object(app.cv2, "VideoCapture", side_effect=lambda index, _backend: cameras[index]):
        camera, first_frame = app.open_camera(0)

    assert camera is working
    assert first_frame is frame
    assert failed_open.released
    assert failed_read.released
    assert not working.released


def test_open_camera_rejects_open_device_that_cannot_return_a_frame():
    opened_no_frame = FakeCamera(reads=[(False, None)])
    with patch.object(app.cv2, "VideoCapture", return_value=opened_no_frame):
        with pytest.raises(RuntimeError, match="Could not obtain a frame"):
            app.open_camera(0)
    assert opened_no_frame.released


def test_open_camera_rejects_black_frame_as_unusable():
    black_frame = np.zeros((8, 10, 3), dtype=np.uint8)
    camera = FakeCamera(reads=[(True, black_frame)])
    with patch.object(app.cv2, "VideoCapture", return_value=camera):
        with pytest.raises(RuntimeError, match="no usable frame"):
            app.open_camera(0)
    assert camera.released


def test_failed_runtime_read_stops_capture_without_processing_or_counting_frame(capsys):
    frame = np.ones((8, 10, 3), dtype=np.uint8)
    camera = FakeCamera(reads=[(False, None)])

    class Extractor:
        last_hand_results = None
        last_presence = {
            "left_hand": False,
            "right_hand": False,
        }

        def __init__(self):
            self.processed = 0
            self.closed = False

        def process_frame(self, _frame):
            self.processed += 1
            return np.zeros(177, dtype=np.float32)

        def draw_annotations(self, image):
            return image

        def close(self):
            self.closed = True

    extractor = Extractor()
    keys = iter([ord("s")])
    with patch.object(sys, "argv", ["webcam.py"]), \
         patch.object(app, "load_artifact", return_value=(object(), object(), [])), \
         patch.object(app, "open_camera", return_value=(camera, frame)), \
         patch.object(app.YoloPoseHandsExtractor, "from_checkpoint", return_value=extractor), \
         patch.object(app, "preprocess_features", return_value=(np.zeros(177), np.zeros(3))), \
         patch.object(app.cv2, "imshow"), \
         patch.object(app.cv2, "waitKey", side_effect=lambda _delay: next(keys)), \
         patch.object(app.cv2, "putText", side_effect=lambda image, *_args: image), \
         patch.object(app.cv2, "destroyAllWindows"):
        app.main()

    assert extractor.processed == 0
    assert extractor.closed
    assert camera.released
    assert "Camera error: Frame read failed or returned a blank image. Capture stopped." in capsys.readouterr().err


def test_lighting_calibration_displays_all_statistics_and_quits_without_model(
    monkeypatch,
):
    frames = [
        np.full((8, 10, 3), 40, dtype=np.uint8),
        np.full((8, 10, 3), 80, dtype=np.uint8),
    ]
    camera = FakeCamera(reads=[(True, frames[1])])
    measurements = {
        "mean_brightness": 80.0,
        "median_brightness": 79.0,
        "std_brightness": 5.0,
        "percent_below_50": 2.0,
        "percent_above_200": 1.0,
    }
    analyzed_frames = []
    rendered_text = []
    keys = iter([0, ord("q")])

    monkeypatch.setattr(app, "open_camera", lambda _index: (camera, frames[0]))
    monkeypatch.setattr(
        app,
        "analyze_lighting",
        lambda frame: analyzed_frames.append(frame) or measurements,
    )
    monkeypatch.setattr(app, "load_artifact", lambda *_args: pytest.fail("model loaded"))
    monkeypatch.setattr(app, "predict_sequence", lambda *_args: pytest.fail("prediction ran"))
    monkeypatch.setattr(app, "append_experiment_record", lambda *_args: pytest.fail("record logged"))
    monkeypatch.setattr("builtins.input", lambda *_args: pytest.fail("input requested"))
    monkeypatch.setattr(app.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda _delay: next(keys))
    monkeypatch.setattr(
        app.cv2,
        "putText",
        lambda image, text, *_args: rendered_text.append(text) or image,
    )
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)

    monkeypatch.setattr(sys, "argv", ["webcam.py", "--lighting-calibration"])
    app.main()

    assert len(analyzed_frames) == 2
    assert all(
        any(key in text for text in rendered_text)
        for key in app.LIGHTING_STAT_KEYS
    )
    assert camera.released


def test_lighting_calibration_releases_camera_after_read_failure(monkeypatch, capsys):
    frame = np.full((8, 10, 3), 40, dtype=np.uint8)
    camera = FakeCamera(reads=[(False, None)])

    monkeypatch.setattr(app, "open_camera", lambda _index: (camera, frame))
    monkeypatch.setattr(app, "analyze_lighting", lambda _frame: {
        key: 1.0 for key in app.LIGHTING_STAT_KEYS
    })
    monkeypatch.setattr(app.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda _delay: 0)
    monkeypatch.setattr(app.cv2, "putText", lambda image, *_args: image)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)

    app.run_lighting_calibration()

    assert camera.released
    assert "Calibration stopped" in capsys.readouterr().err


def _run_mocked_lighting_experiment(monkeypatch, output_path, missing_at=None):
    frame = np.ones((8, 10, 3), dtype=np.uint8)

    class StreamingCamera:
        released = False

        def read(self):
            return True, frame.copy()

        def release(self):
            self.released = True

    class Extractor:
        last_hand_results = None
        last_presence = {"left_hand": False, "right_hand": False}

        def process_frame(self, _frame):
            return np.zeros(177, dtype=np.float32)

        def draw_annotations(self, image):
            return image

        def close(self):
            pass

    camera = StreamingCamera()
    extractor = Extractor()
    labels = [f"sign-{index}" for index in range(100)]
    lighting = {
        "mean_brightness": 92.5,
        "median_brightness": 90.0,
        "std_brightness": 18.25,
        "percent_below_50": 7.5,
        "percent_above_200": 2.25,
    }
    analyzed_count = 0
    predictions = []
    keys = iter([ord("s")] + [0] * 29 + [ord(" "), 13, ord("q")])
    user_inputs = iter(["P07", "EXP12", "uneven", "water"])

    def analyze_frame(_frame):
        nonlocal analyzed_count
        analyzed_count += 1
        if analyzed_count == missing_at:
            return None
        return lighting.copy()

    def predict(_classifier, _normalizer, sequence):
        predictions.append(sequence.copy())
        return 28, 0.91

    monkeypatch.setattr(sys, "argv", ["webcam.py", "--lighting-experiment"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(user_inputs))
    monkeypatch.setattr(app, "load_artifact", lambda _path: (object(), object(), labels))
    monkeypatch.setattr(app, "open_camera", lambda _index: (camera, frame))
    monkeypatch.setattr(
        app.YoloPoseHandsExtractor,
        "from_checkpoint",
        lambda _path: extractor,
    )
    monkeypatch.setattr(
        app,
        "preprocess_features",
        lambda *_args: (np.zeros(177), np.zeros(3)),
    )
    monkeypatch.setattr(app, "draw_hand_diagnostics", lambda *_args: None)
    monkeypatch.setattr(app, "analyze_lighting", analyze_frame)
    monkeypatch.setattr(app, "predict_sequence", predict)
    monkeypatch.setattr(app.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda _delay: next(keys))
    monkeypatch.setattr(app.cv2, "putText", lambda image, *_args: image)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(
        app,
        "append_experiment_record",
        lambda record: append_experiment_record(record, output_path),
    )

    app.main()

    return camera, analyzed_count, predictions


def test_lighting_experiment_main_writes_one_complete_prediction_row(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "synthetic_lighting_results.csv"

    camera, analyzed_count, predictions = _run_mocked_lighting_experiment(
        monkeypatch, output_path
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert camera.released
    assert analyzed_count == 30
    assert len(predictions) == 1
    assert len(rows) == 1
    assert rows[0]["predicted_sign"] == "sign-28"
    assert rows[0]["actual_sign"] == "water"
    assert rows[0]["participant_id"] == "P07"
    assert rows[0]["experiment_id"] == "EXP12"
    assert rows[0]["lighting_condition"] == "uneven"
    assert {
        key: float(rows[0][key])
        for key in app.LIGHTING_STAT_KEYS
    } == {
        "mean_brightness": 92.5,
        "median_brightness": 90.0,
        "std_brightness": 18.25,
        "percent_below_50": 7.5,
        "percent_above_200": 2.25,
    }


def test_lighting_experiment_main_skips_row_for_incomplete_interval(
    monkeypatch, tmp_path
):
    output_path = tmp_path / "incomplete_lighting_results.csv"

    camera, analyzed_count, predictions = _run_mocked_lighting_experiment(
        monkeypatch, output_path, missing_at=12
    )

    assert camera.released
    assert analyzed_count == 30
    assert len(predictions) == 1
    assert not output_path.exists()

