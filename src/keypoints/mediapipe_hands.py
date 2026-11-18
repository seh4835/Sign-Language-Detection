"""MediaPipe Hands extraction isolated from the existing Holistic pipeline."""

from __future__ import annotations

from typing import Any

import mediapipe as mp
import numpy as np


class MediaPipeHandsExtractor:
    """Extract independently detected left and right hand landmarks.

    The output follows the existing HolisticExtractor hand convention:
    missing hands are zero-filled and each hand contains 21 ``(x, y, z)``
    landmarks. Combined features are ordered left hand, then right hand.
    """

    HAND_LANDMARKS = 21
    HAND_FEATURE_DIM = HAND_LANDMARKS * 3
    COMBINED_FEATURE_DIM = HAND_FEATURE_DIM * 2

    def __init__(
        self,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if max_num_hands != 2:
            raise ValueError("This adapter requires max_num_hands=2.")

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.last_results: Any | None = None

    @staticmethod
    def _extract_hand(hand_landmarks: Any | None) -> np.ndarray:
        """Return one hand as ``(21, 3)`` or the established zero-fill."""
        if hand_landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)

        landmarks = np.asarray(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark],
            dtype=np.float32,
        )
        if landmarks.shape != (21, 3):
            raise ValueError(
                f"Expected MediaPipe hand shape (21, 3), got {landmarks.shape}."
            )
        return landmarks

    def process_frame(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process one RGB frame and return left, right, and combined features."""
        self.last_results = self.hands.process(frame_rgb)
        left_hand = None
        right_hand = None

        multi_hand_landmarks = self.last_results.multi_hand_landmarks or []
        handedness = self.last_results.multi_handedness or []
        if len(multi_hand_landmarks) != len(handedness):
            raise RuntimeError(
                "MediaPipe Hands returned mismatched landmarks and handedness results."
            )

        for landmarks, handedness_result in zip(
            multi_hand_landmarks, handedness
        ):
            if not handedness_result.classification:
                raise RuntimeError("MediaPipe Hands returned empty handedness data.")
            label = handedness_result.classification[0].label
            if label == "Left":
                if left_hand is not None:
                    raise RuntimeError("MediaPipe Hands returned duplicate left hands.")
                left_hand = landmarks
            elif label == "Right":
                if right_hand is not None:
                    raise RuntimeError("MediaPipe Hands returned duplicate right hands.")
                right_hand = landmarks
            else:
                raise RuntimeError(f"Unexpected MediaPipe hand label: {label!r}.")

        left = self._extract_hand(left_hand)
        right = self._extract_hand(right_hand)
        combined = np.concatenate((left.reshape(-1), right.reshape(-1))).astype(
            np.float32
        )
        if combined.shape != (self.COMBINED_FEATURE_DIM,):
            raise ValueError(
                f"Expected combined hand shape ({self.COMBINED_FEATURE_DIM},), "
                f"got {combined.shape}."
            )
        return left, right, combined

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw the latest detected hand landmarks on a BGR frame."""
        if self.last_results is None:
            return frame_bgr

        output = frame_bgr.copy()
        for hand_landmarks in self.last_results.multi_hand_landmarks or []:
            self.mp_drawing.draw_landmarks(
                output,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
            )
        return output

    def close(self) -> None:
        self.hands.close()
