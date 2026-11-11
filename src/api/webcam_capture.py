"""Standalone webcam capture for the YOLO11 + MediaPipe Hands pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.keypoints.yolo_pose_hands import (
    YoloPoseHandsExtractor,
    preprocess_features,
)


FRAME_COUNT = 30


def sample_uniform_frames(frames: list[np.ndarray], count: int = FRAME_COUNT) -> list[np.ndarray]:
    """Uniformly sample ``count`` frames across a captured interval."""
    if count <= 0:
        raise ValueError("count must be positive.")
    if len(frames) < count:
        raise IncompleteCaptureError(
            f"Capture interval has only {len(frames)}/{count} usable frames."
        )
    indices = np.linspace(0, len(frames) - 1, count, dtype=np.int64)
    return [frames[int(index)] for index in indices]


class CameraUnavailableError(RuntimeError):
    """Raised when no requested webcam can be opened."""


class IncompleteCaptureError(RuntimeError):
    """Raised when a sign does not produce 30 usable chronological frames."""


class CaptureCancelledError(RuntimeError):
    """Raised when the user cancels an active capture."""


@dataclass(frozen=True)
class CaptureResult:
    """Raw, presence-aware, and LSTM-ready output for one captured sign."""

    raw_features: np.ndarray
    presence: np.ndarray
    lstm_features: np.ndarray

    def __post_init__(self) -> None:
        if self.raw_features.shape != (FRAME_COUNT, 177):
            raise ValueError(f"Expected raw shape (30, 177), got {self.raw_features.shape}.")
        if self.presence.shape != (FRAME_COUNT, 3):
            raise ValueError(f"Expected presence shape (30, 3), got {self.presence.shape}.")
        if self.lstm_features.shape != (FRAME_COUNT, 180):
            raise ValueError(f"Expected LSTM shape (30, 180), got {self.lstm_features.shape}.")
        if not (
            np.isfinite(self.raw_features).all()
            and np.isfinite(self.presence).all()
            and np.isfinite(self.lstm_features).all()
        ):
            raise ValueError("Capture result contains non-finite values.")


class WebcamCapture:
    """Capture exactly 30 usable chronological frames from a webcam.

    Press ``S`` in the live preview to start, ``Q`` to cancel/quit, and
    ``R`` after a completed capture to discard it and capture again.
    """

    def __init__(
        self,
        extractor: Any,
        capture: Any,
        *,
        window_name: str = "YOLO11 Pose + MediaPipe Hands",
    ) -> None:
        self.extractor = extractor
        self.capture = capture
        self.window_name = window_name
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        camera_index: int = 0,
        window_name: str = "YOLO11 Pose + MediaPipe Hands",
    ) -> "WebcamCapture":
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            raise CameraUnavailableError(
                f"Could not open webcam index {camera_index}. "
                "Check camera permissions, connection, and whether another "
                "application is using the camera."
            )
        try:
            extractor = YoloPoseHandsExtractor.from_checkpoint(str(checkpoint))
        except Exception:
            capture.release()
            raise
        return cls(extractor, capture, window_name=window_name)

    @staticmethod
    def _presence_array(presence: dict[str, bool]) -> np.ndarray:
        return np.asarray(
            [
                bool(presence["body_pose"]),
                bool(presence["left_hand"]),
                bool(presence["right_hand"]),
            ],
            dtype=np.float32,
        )

    def capture_one_sign(self, *, display: bool = True) -> CaptureResult:
        """Collect 30 successful frames in read order and preprocess them."""
        return self._capture_frames(FRAME_COUNT, display=display, stop_on_key=False)

    def capture_interval(self, *, display: bool = True) -> CaptureResult:
        """Capture a complete sign interval, then sample it uniformly."""
        return self._capture_frames(None, display=display, stop_on_key=True)

    def _capture_frames(
        self,
        target_count: int | None,
        *,
        display: bool,
        stop_on_key: bool,
    ) -> CaptureResult:
        raw_frames: list[np.ndarray] = []
        presence_frames: list[np.ndarray] = []
        width = height = 0

        while target_count is None or len(raw_frames) < target_count:
            success, frame = self.capture.read()
            if not success or frame is None:
                raise IncompleteCaptureError(
                    f"Capture ended after {len(raw_frames)}/{FRAME_COUNT} usable frames."
                )
            if frame.ndim < 2:
                raise IncompleteCaptureError("Camera returned an invalid frame.")
            height, width = frame.shape[:2]
            raw = self.extractor.process_frame(frame)
            if raw.shape != (177,) or not np.isfinite(raw).all():
                raise ValueError(f"Extractor returned invalid frame features: {raw.shape}.")
            presence = self._presence_array(self.extractor.last_presence)
            processed, indicators = preprocess_features(
                raw, width, height, presence
            )
            raw_frames.append(raw.astype(np.float32, copy=True))
            presence_frames.append(indicators.astype(np.float32, copy=True))

            if display:
                annotated = self.extractor.draw_annotations(frame)
                cv2.putText(
                    annotated,
                    f"Capturing: {len(raw_frames)} frames (Space=stop)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(self.window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    raise CaptureCancelledError("Capture cancelled by user.")
                if stop_on_key and key in (ord(" "), 13, 10):
                    break

        selected_raw = sample_uniform_frames(raw_frames) if stop_on_key else raw_frames
        selected_presence = (
            sample_uniform_frames(presence_frames) if stop_on_key else presence_frames
        )

        raw_sequence = np.asarray(selected_raw, dtype=np.float32)
        presence_sequence = np.asarray(selected_presence, dtype=np.float32)
        processed_frames = [
            preprocess_features(raw, width, height, indicators)[0]
            for raw, indicators in zip(raw_sequence, presence_sequence)
        ]
        normalized = np.asarray(processed_frames, dtype=np.float32)
        lstm_sequence = np.concatenate((normalized, presence_sequence), axis=1)
        return CaptureResult(raw_sequence, presence_sequence, lstm_sequence)

    def run(self) -> CaptureResult | None:
        """Run the interactive preview and return an accepted capture.

        ``None`` means the user quit before accepting a capture.
        """
        if not self.capture.isOpened():
            raise CameraUnavailableError("The webcam is not open.")
        while True:
            success, frame = self.capture.read()
            if not success:
                raise CameraUnavailableError("Could not read a live webcam frame.")
            display = frame.copy()
            cv2.putText(
                display,
                "S=start capture  Q=quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key == ord("s"):
                try:
                    result = self.capture_one_sign(display=True)
                except CaptureCancelledError:
                    continue
                while True:
                    accepted = self._wait_for_capture_decision()
                    if accepted:
                        return result
                    if accepted is None:
                        return None
                    break

    def _wait_for_capture_decision(self) -> bool | None:
        while True:
            display = np.zeros((120, 720, 3), dtype=np.uint8)
            cv2.putText(
                display,
                "Enter=accept  R=recapture  Q=quit",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):
                return True
            if key == ord("r"):
                return False
            if key in (ord("q"), 27):
                return None

    def close(self) -> None:
        if not self._closed:
            self.capture.release()
            self.extractor.close()
            cv2.destroyAllWindows()
            self._closed = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"outputs\yolo11_pose_test\yolo11n-pose.pt"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = WebcamCapture.from_checkpoint(args.model, camera_index=args.camera_index)
    try:
        result = session.run()
        if result is None:
            print("Capture cancelled.")
            return 0
        print(
            "Capture complete: "
            f"raw={result.raw_features.shape}, "
            f"presence={result.presence.shape}, "
            f"lstm={result.lstm_features.shape}"
        )
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
