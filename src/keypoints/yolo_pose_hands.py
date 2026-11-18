"""Combined YOLO11 pose and MediaPipe Hands feature extraction."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

PRESENCE_NAMES = ("body_pose", "left_hand", "right_hand")


def preprocess_features(
    features: np.ndarray,
    frame_width: int,
    frame_height: int,
    presence: dict[str, bool] | np.ndarray,
    *,
    hand_z_clip: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one raw ``(177,)`` frame and return indicators separately.

    YOLO body x/y pixel coordinates are divided by frame width/height.
    YOLO confidence and MediaPipe hand x/y are already dimensionless and are
    unchanged. MediaPipe z is relative depth, so it is clipped symmetrically
    to ``[-hand_z_clip, hand_z_clip]`` to bound its influence without treating
    it as confidence. Zero-filled missing groups stay zero. The returned
    indicators have order ``body_pose, left_hand, right_hand``; an LSTM
    pipeline should concatenate them as three auxiliary values (177 + 3)
    or provide them through a separate input.
    """
    raw = np.asarray(features, dtype=np.float32)
    if raw.shape != (YoloPoseHandsExtractor.FEATURE_DIM,):
        raise ValueError(f"Expected raw feature shape (177,), got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("Raw features contain non-finite values.")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if hand_z_clip <= 0 or not np.isfinite(hand_z_clip):
        raise ValueError("hand_z_clip must be finite and positive.")

    if isinstance(presence, dict):
        missing = set(PRESENCE_NAMES) - set(presence)
        if missing:
            raise ValueError(f"Missing presence indicators: {sorted(missing)}.")
        indicators = np.asarray([bool(presence[name]) for name in PRESENCE_NAMES], dtype=np.float32)
    else:
        indicators = np.asarray(presence, dtype=np.float32)
        if indicators.shape != (3,) or not np.isin(indicators, (0.0, 1.0)).all():
            raise ValueError("Presence indicators must be three binary values.")

    processed = raw.copy()
    processed[0:51:3] /= np.float32(frame_width)
    processed[1:51:3] /= np.float32(frame_height)
    processed[116:177:3] = np.clip(processed[116:177:3], -hand_z_clip, hand_z_clip)
    processed[53:114:3] = np.clip(processed[53:114:3], -hand_z_clip, hand_z_clip)
    if indicators[0] == 0:
        processed[:51] = 0.0
    if indicators[1] == 0:
        processed[51:114] = 0.0
    if indicators[2] == 0:
        processed[114:177] = 0.0
    if not np.isfinite(processed).all():
        raise ValueError("Preprocessed features contain non-finite values.")
    return processed, indicators


class YoloPoseHandsExtractor:
    """Extract one fixed-size feature vector from each BGR video frame.

    Layout (177 float32 values):
    ``[body pose (17 * [x_pixel, y_pixel, confidence]),
    left hand (21 * [x, y, z]), right hand (21 * [x, y, z])]``.

    Missing body or hands are zero-filled. Detection presence is deliberately
    kept out of the feature vector and exposed through ``last_presence`` so
    the feature dimension remains exactly 177.
    """

    BODY_KEYPOINTS = 17
    BODY_FEATURE_DIM = BODY_KEYPOINTS * 3
    HAND_LANDMARKS = 21
    HAND_FEATURE_DIM = HAND_LANDMARKS * 3
    FEATURE_DIM = BODY_FEATURE_DIM + 2 * HAND_FEATURE_DIM

    def __init__(
        self,
        model: Any,
        *,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if max_num_hands != 2:
            raise ValueError("This adapter requires max_num_hands=2.")
        self.model = model
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_pose_result: Any | None = None
        self.last_hand_results: Any | None = None
        self.last_presence = {
            "body_pose": False,
            "left_hand": False,
            "right_hand": False,
        }
        self.last_hand_warnings: list[str] = []

    @classmethod
    def from_checkpoint(cls, checkpoint: str, **kwargs: Any) -> "YoloPoseHandsExtractor":
        """Create an extractor from a YOLO checkpoint or model name."""
        return cls(YOLO(checkpoint), **kwargs)

    @staticmethod
    def _zero_body() -> np.ndarray:
        return np.zeros((YoloPoseHandsExtractor.BODY_KEYPOINTS, 3), dtype=np.float32)

    @staticmethod
    def _zero_hand() -> np.ndarray:
        return np.zeros((YoloPoseHandsExtractor.HAND_LANDMARKS, 3), dtype=np.float32)

    @staticmethod
    def _extract_body(result: Any) -> tuple[np.ndarray, bool]:
        if result.keypoints is None or result.keypoints.data.shape[0] == 0:
            return YoloPoseHandsExtractor._zero_body(), False
        body = result.keypoints.data[0].detach().cpu().numpy().astype(np.float32)
        if body.shape != (YoloPoseHandsExtractor.BODY_KEYPOINTS, 3):
            raise ValueError(f"Expected YOLO body shape (17, 3), got {body.shape}.")
        if not np.isfinite(body).all():
            raise ValueError("YOLO body output contains non-finite values.")
        if ((body[:, 2] < 0) | (body[:, 2] > 1)).any():
            raise ValueError("YOLO body confidence values must be in [0, 1].")
        return body, True

    @staticmethod
    def _extract_hand(hand_landmarks: Any | None) -> tuple[np.ndarray, bool]:
        if hand_landmarks is None:
            return YoloPoseHandsExtractor._zero_hand(), False
        hand = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
            dtype=np.float32,
        )
        if hand.shape != (YoloPoseHandsExtractor.HAND_LANDMARKS, 3):
            raise ValueError(f"Expected MediaPipe hand shape (21, 3), got {hand.shape}.")
        if not np.isfinite(hand).all():
            raise ValueError("MediaPipe hand output contains non-finite values.")
        return hand, True

    @staticmethod
    def combine_features(body: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Combine arrays in the documented body, left-hand, right-hand order."""
        if body.shape != (17, 3) or left.shape != (21, 3) or right.shape != (21, 3):
            raise ValueError("Unexpected body or hand shape.")
        features = np.concatenate((body.reshape(-1), left.reshape(-1), right.reshape(-1)))
        if features.shape != (YoloPoseHandsExtractor.FEATURE_DIM,):
            raise ValueError(f"Expected feature shape (177,), got {features.shape}.")
        if not np.isfinite(features).all():
            raise ValueError("Combined features contain non-finite values.")
        return features.astype(np.float32, copy=False)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return exactly one ``(177,)`` feature vector for a BGR frame."""
        pose_results = self.model.predict(source=frame_bgr, verbose=False)
        if not pose_results:
            raise RuntimeError("YOLO returned no result for the frame.")
        self.last_pose_result = pose_results[0]
        body, body_present = self._extract_body(self.last_pose_result)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.last_hand_results = self.hands.process(frame_rgb)
        self.last_hand_warnings = []
        left_source = right_source = None
        landmarks = self.last_hand_results.multi_hand_landmarks or []
        handedness = self.last_hand_results.multi_handedness or []
        if len(landmarks) != len(handedness):
            raise RuntimeError("MediaPipe Hands returned mismatched results.")
        for hand, handedness_result in zip(landmarks, handedness):
            if not handedness_result.classification:
                raise RuntimeError("MediaPipe Hands returned empty handedness data.")
            label = handedness_result.classification[0].label
            if label == "Left":
                if left_source is not None:
                    self.last_hand_warnings.append("duplicate_left_handedness")
                    continue
                left_source = hand
            elif label == "Right":
                if right_source is not None:
                    self.last_hand_warnings.append("duplicate_right_handedness")
                    continue
                right_source = hand
            else:
                raise RuntimeError(f"Unexpected MediaPipe hand label: {label!r}.")

        left, left_present = self._extract_hand(left_source)
        right, right_present = self._extract_hand(right_source)
        self.last_presence = {
            "body_pose": body_present,
            "left_hand": left_present,
            "right_hand": right_present,
        }
        return self.combine_features(body, left, right)

    def draw_annotations(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw the latest YOLO and hand detections on a BGR frame."""
        output = (
            self.last_pose_result.plot()
            if self.last_pose_result is not None
            else frame_bgr.copy()
        )
        if self.last_hand_results is not None:
            for hand in self.last_hand_results.multi_hand_landmarks or []:
                self.mp_drawing.draw_landmarks(output, hand, self.mp_hands.HAND_CONNECTIONS)
        return output

    def close(self) -> None:
        self.hands.close()
