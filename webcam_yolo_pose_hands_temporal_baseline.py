
"""Live YOLO11 + MediaPipe Hands temporal baseline sign prediction."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from src.keypoints.lighting_analysis import analyze_lighting
from scripts.log_lighting_experiment import (
    append_experiment_record,
    validate_experiment_ids,
)
from src.keypoints.yolo_pose_hands import (
    YoloPoseHandsExtractor,
    preprocess_features,
)

from src.models.temporal_baseline import (
    load_artifact,
    predict_sequence,
    sample_uniform_sequence,
)


ROOT = Path(__file__).resolve().parent

ARTIFACTS = (
    ROOT
    / "models"
    / "yolo_pose_hands_lstm_v2"
    / "webcam_baseline_artifacts"
)

WINDOW_NAME = "YOLO11 + MediaPipe Hands Temporal Baseline"
LIGHTING_STAT_KEYS = (
    "mean_brightness",
    "median_brightness",
    "std_brightness",
    "percent_below_50",
    "percent_above_200",
)


def _usable_frame(frame: np.ndarray | None) -> bool:
    """Check whether a camera frame is valid."""
    return (
        frame is not None
        and frame.size > 0
        and bool(np.any(frame))
    )


def aggregate_lighting_statistics(
    statistics: list[Mapping[str, float] | None],
) -> dict[str, float] | None:
    """Return means for valid per-frame lighting statistics."""
    valid: list[list[float]] = []
    for measurement in statistics:
        if measurement is None:
            continue
        try:
            values = [float(measurement[key]) for key in LIGHTING_STAT_KEYS]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(values).all():
            valid.append(values)

    if not valid:
        return None

    means = np.mean(np.asarray(valid, dtype=np.float64), axis=0)
    return dict(zip(LIGHTING_STAT_KEYS, means.tolist()))


def aggregate_complete_lighting_interval(
    statistics: list[Mapping[str, float] | None],
    expected_count: int,
) -> tuple[dict[str, float] | None, int]:
    """Aggregate only when every expected frame has valid lighting data."""
    valid_count = 0
    for measurement in statistics:
        if measurement is None:
            continue
        try:
            values = [float(measurement[key]) for key in LIGHTING_STAT_KEYS]
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(values).all():
            valid_count += 1

    missing_count = max(0, expected_count - valid_count)
    if len(statistics) != expected_count:
        missing_count = max(missing_count, abs(expected_count - len(statistics)))
    if missing_count or valid_count != expected_count:
        return None, max(missing_count, 1)
    return aggregate_lighting_statistics(statistics), 0


def try_log_experiment_record(
    record: Mapping[str, object],
) -> str | None:
    """Append one record and return a clear error instead of raising."""
    try:
        append_experiment_record(record)
    except Exception as exc:
        return f"Experiment CSV logging failed; prediction remains available: {exc}"
    return None


def build_lighting_experiment_record(
    *,
    experiment_id: str,
    participant_id: str,
    lighting_condition: str,
    predicted_sign: str,
    actual_sign: str,
    lighting_statistics: Mapping[str, float],
) -> dict[str, object]:
    """Build a scalar-only record accepted by the existing experiment logger."""
    return {
        "experiment_id": experiment_id,
        "participant_id": participant_id,
        "lighting_condition": lighting_condition,
        **{key: float(lighting_statistics[key]) for key in LIGHTING_STAT_KEYS},
        "predicted_sign": predicted_sign,
        "actual_sign": actual_sign,
    }


def open_camera(
    requested_index: int,
) -> tuple[cv2.VideoCapture, np.ndarray]:
    """Open the first requested or standard camera that returns a real frame."""

    backends = (
        [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        if sys.platform == "win32"
        else [cv2.CAP_ANY]
    )

    candidates = list(dict.fromkeys([requested_index, 0, 1, 2]))
    failures: list[str] = []

    for index in candidates:
        for backend in backends:

            camera = cv2.VideoCapture(index, backend)

            if not camera.isOpened():
                failures.append(
                    f"Camera {index} did not open with backend {backend}"
                )
                camera.release()
                continue

            ok, frame = camera.read()

            if not ok or not _usable_frame(frame):
                failures.append(
                    f"Camera {index} opened but returned no usable frame"
                )
                camera.release()
                continue

            print(
                f"Camera {index} opened successfully. "
                f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
            )

            return camera, frame

    detail = "; ".join(failures)

    raise RuntimeError(
        f"Could not obtain a frame from camera indices {candidates}. "
        f"{detail}. Check that the webcam is connected and available."
    )


def run_lighting_calibration(camera_index: int = 0) -> None:
    """Display live lighting statistics without loading or using a model."""
    try:
        camera, frame = open_camera(camera_index)
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return

    try:
        while True:
            display = frame.copy()
            try:
                statistics = analyze_lighting(frame)
                lines = [
                    f"{key}: {statistics[key]:.2f}"
                    for key in LIGHTING_STAT_KEYS
                ]
            except (TypeError, ValueError):
                lines = ["Lighting measurements unavailable"]

            cv2.putText(
                display,
                "Lighting calibration | Q = quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            for index, line in enumerate(lines):
                cv2.putText(
                    display,
                    line,
                    (20, 70 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 230, 20),
                    2,
                )

            cv2.imshow("Lighting Calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            ok, next_frame = camera.read()
            if not ok or not _usable_frame(next_frame):
                print(
                    "Camera error: Frame read failed or returned a blank image. "
                    "Calibration stopped.",
                    file=sys.stderr,
                )
                break
            frame = next_frame
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Lighting calibration camera shut down cleanly.")


def draw_hand_diagnostics(
    display: np.ndarray,
    extractor: YoloPoseHandsExtractor,
    capturing: bool,
) -> None:
    """Display live MediaPipe hand detection information."""

    if extractor.last_hand_results is not None:
        landmarks = (
            extractor.last_hand_results.multi_hand_landmarks or []
        )
        hand_count = len(landmarks)
    else:
        hand_count = 0

    presence = extractor.last_presence

    left_detected = presence["left_hand"]
    right_detected = presence["right_hand"]

    if not capturing:
        status = "Hand diagnostics: Start capture to monitor"
        color = (255, 255, 255)
    else:
        status = (
            f"Hands: {hand_count} | "
            f"Left: {'YES' if left_detected else 'NO'} | "
            f"Right: {'YES' if right_detected else 'NO'}"
        )

        if hand_count == 2:
            color = (0, 255, 0)
        elif hand_count == 1:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

    cv2.putText(
        display,
        status,
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pose-model",
        type=Path,
        default=ROOT / "outputs/yolo11_pose_test/yolo11n-pose.pt",
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ARTIFACTS,
    )

    parser.add_argument(
        "--camera-probe",
        action="store_true",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--lighting-experiment",
        action="store_true",
        help="Prompt for experiment metadata and log each prediction attempt.",
    )

    parser.add_argument(
        "--lighting-calibration",
        action="store_true",
        help="Display live lighting statistics without running predictions.",
    )

    args = parser.parse_args()

    if args.lighting_calibration:
        run_lighting_calibration(args.camera)
        return

    # Load the existing trained classifier and normalization artifacts.
    classifier, normalizer, labels = load_artifact(args.artifacts)

    if args.self_test:

        label, confidence = predict_sequence(
            classifier,
            normalizer,
            np.zeros((30, 180), dtype=np.float32),
        )

        print(
            f"Self-test prediction: "
            f"{labels[label]} ({confidence:.1%})"
        )

        return

    experiment_metadata: dict[str, str] | None = None
    if args.lighting_experiment:
        participant_id = input("Anonymous participant ID: ").strip()
        experiment_id = input("Experiment ID: ").strip()
        lighting_condition = input(
            "Lighting condition (bright/normal/dim/uneven): "
        ).strip().lower()
        try:
            validate_experiment_ids(participant_id, experiment_id)
        except ValueError as exc:
            parser.error(str(exc))
        if lighting_condition not in {"bright", "normal", "dim", "uneven"}:
            parser.error(
                "Lighting condition must be bright, normal, dim, or uneven."
            )
        experiment_metadata = {
            "participant_id": participant_id,
            "experiment_id": experiment_id,
            "lighting_condition": lighting_condition,
        }

    # Open webcam.
    try:
        camera, first_frame = open_camera(args.camera)

    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return

    if args.camera_probe:
        camera.release()
        print("Camera probe released cleanly.")
        return

    extractor = None

    interval: list[np.ndarray] = []
    lighting_interval: list[dict[str, float] | None] = []

    result = (
        "Press S to start; Space to predict; R to reset",
        0.0,
    )

    diagnostics: list[str] | None = None

    capturing = False

    try:

        # Initialize YOLO11 Pose + MediaPipe Hands.
        extractor = YoloPoseHandsExtractor.from_checkpoint(
            str(args.pose_model)
        )

        print(
            "Camera ready.\n"
            "S = Start capture\n"
            "Space / Enter = Predict\n"
            "R = Reset\n"
            "Q / Esc = Quit"
        )

        frame = first_frame

        while True:

            if capturing:

                # Extract YOLO11 body and MediaPipe hand features.
                raw = extractor.process_frame(frame)

                processed, presence = preprocess_features(
                    raw,
                    frame.shape[1],
                    frame.shape[0],
                    extractor.last_presence,
                )

                # Maintain the original 180-feature frame representation.
                frame_features = np.concatenate(
                    (processed, presence)
                ).astype(np.float32)

                interval.append(frame_features)
                try:
                    lighting_interval.append(analyze_lighting(frame))
                except (TypeError, ValueError):
                    lighting_interval.append(None)

                # Draw YOLO11 and MediaPipe annotations.
                display = extractor.draw_annotations(frame)

            else:

                display = frame.copy()

            # Prediction result.
            if result[1] > 0:
                result_text = f"{result[0]}  {result[1]:.1%}"
            else:
                result_text = result[0]

            cv2.putText(
                display,
                result_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 230, 20),
                2,
            )

            # Capture status.
            capture_text = (
                f"{'Capturing' if capturing else 'Ready'}: "
                f"{len(interval)} frames"
            )

            cv2.putText(
                display,
                capture_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if diagnostics is not None:
                for index, line in enumerate(diagnostics):
                    cv2.putText(
                        display,
                        line,
                        (20, 100 + index * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

            # Live hand detection diagnostics.
            if extractor is not None:
                draw_hand_diagnostics(
                    display,
                    extractor,
                    capturing,
                )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            # Quit.
            if key in (ord("q"), 27):
                break

            # Start capture.
            if key == ord("s"):

                interval = []
                lighting_interval = []
                capturing = True
                diagnostics = None

                result = ("Capture started", 0.0)

                print("Capture started.")

            # Reset capture.
            elif key == ord("r"):

                interval = []
                lighting_interval = []
                capturing = False
                diagnostics = None

                result = ("Capture reset", 0.0)

                print("Capture reset.")

            # Stop capture and predict.
            elif key in (ord(" "), 13, 10) and capturing:

                if len(interval) < 30:

                    result = (
                        f"Need at least 30 frames (got {len(interval)})",
                        0.0,
                    )

                else:

                    # Uniformly sample 30 frames.
                    sampled = sample_uniform_sequence(
                        np.stack(interval)
                    )

                    if sampled.shape != (30, 180):
                        raise ValueError(
                            f"Expected sequence shape (30, 180), "
                            f"got {sampled.shape}"
                        )

                    if experiment_metadata is not None:
                        # Experiment measurements cover every frame in the
                        # complete capture interval, independent of inference
                        # sampling.
                        aggregate_lighting, missing_lighting_count = (
                            aggregate_complete_lighting_interval(
                                lighting_interval,
                                expected_count=len(interval),
                            )
                        )
                    else:
                        # Preserve the normal-mode diagnostics behavior.
                        sampled_indices = np.linspace(
                            0,
                            len(lighting_interval) - 1,
                            len(sampled),
                            dtype=np.int64,
                        )
                        sampled_lighting = [
                            lighting_interval[index]
                            for index in sampled_indices
                        ]
                        aggregate_lighting = aggregate_lighting_statistics(
                            sampled_lighting
                        )

                    sampled_presence = sampled[:, 177:]
                    both_hands = int(
                        np.sum(
                            (sampled_presence[:, 1] == 1)
                            & (sampled_presence[:, 2] == 1)
                        )
                    )
                    left_only = int(
                        np.sum(
                            (sampled_presence[:, 1] == 1)
                            & (sampled_presence[:, 2] == 0)
                        )
                    )
                    right_only = int(
                        np.sum(
                            (sampled_presence[:, 1] == 0)
                            & (sampled_presence[:, 2] == 1)
                        )
                    )
                    neither_hand = int(
                        np.sum(
                            (sampled_presence[:, 1] == 0)
                            & (sampled_presence[:, 2] == 0)
                        )
                    )

                    # Predict using the existing baseline classifier.
                    label, confidence = predict_sequence(
                        classifier,
                        normalizer,
                        sampled,
                    )

                    diagnostics = [
                        (
                            f"Captured: {len(interval)} | "
                            f"Sampled: {len(sampled)}"
                        ),
                        (
                            f"Hands - Both: {both_hands} | "
                            f"Left only: {left_only} | "
                            f"Right only: {right_only} | "
                            f"Neither: {neither_hand}"
                        ),
                    ]
                    if aggregate_lighting is not None:
                        diagnostics.extend(
                            [
                                (
                                    "Lighting mean/median/std: "
                                    f"{aggregate_lighting['mean_brightness']:.1f}/"
                                    f"{aggregate_lighting['median_brightness']:.1f}/"
                                    f"{aggregate_lighting['std_brightness']:.1f}"
                                ),
                                (
                                    "Lighting pixels <50/>200: "
                                    f"{aggregate_lighting['percent_below_50']:.1f}%/"
                                    f"{aggregate_lighting['percent_above_200']:.1f}%"
                                ),
                            ]
                        )
                    missing_hand_frames = len(sampled) - both_hands
                    if missing_hand_frames > 0.2 * len(sampled):
                        diagnostics.append(
                            "WARNING: A hand was missing in more than 20% "
                            f"of sampled frames ({missing_hand_frames}/{len(sampled)})."
                        )

                    result = (
                        labels[label],
                        confidence,
                    )

                    print(
                        f"Prediction: {labels[label]} "
                        f"| Confidence: {confidence:.2%}"
                    )

                    if experiment_metadata is not None:
                        actual_sign = input(
                            "Actual sign label for this attempt: "
                        ).strip()
                        if not actual_sign:
                            print(
                                "Experiment record not logged: actual sign "
                                "label was empty.",
                                file=sys.stderr,
                            )
                        elif aggregate_lighting is None:
                            print(
                                "Experiment record not logged: "
                                f"{missing_lighting_count} of {len(interval)} "
                                "frame lighting measurements were missing or invalid.",
                                file=sys.stderr,
                            )
                        else:
                            record = build_lighting_experiment_record(
                                **experiment_metadata,
                                predicted_sign=labels[label],
                                actual_sign=actual_sign,
                                lighting_statistics=aggregate_lighting,
                            )
                            logging_error = try_log_experiment_record(record)
                            if logging_error is not None:
                                print(
                                    logging_error,
                                    file=sys.stderr,
                                )

                capturing = False

            # Read next webcam frame.
            ok, next_frame = camera.read()

            if not ok or not _usable_frame(next_frame):

                capturing = False
                interval = []
                lighting_interval = []

                print(
                    "Camera error: Frame read failed or returned "
                    "a blank image. Capture stopped.",
                    file=sys.stderr,
                )

                break

            frame = next_frame

    finally:

        camera.release()

        if extractor is not None:
            extractor.close()

        cv2.destroyAllWindows()

        print(
            "Camera, extractor, and display shut down cleanly."
        )


if __name__ == "__main__":
    main()
