
import numpy as np
import mediapipe as mp


class HolisticExtractor:
    """
    Extracts MediaPipe Holistic landmarks from an RGB frame.

    Feature layout:
        Face:       468 x 3
        Pose:        33 x 4
        Left hand:   21 x 3
        Right hand:  21 x 3

    Total feature dimension = 1662
    """

    FACE_LANDMARKS = 468
    POSE_LANDMARKS = 33
    HAND_LANDMARKS = 21

    FEATURE_DIM = 1662

    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.last_results = None

    @staticmethod
    def _extract_face(face_landmarks):
        if face_landmarks is None:
            return np.zeros((468, 3), dtype=np.float32)

        return np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

    @staticmethod
    def _extract_pose(pose_landmarks):
        if pose_landmarks is None:
            return np.zeros((33, 4), dtype=np.float32)

        return np.array(
            [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _extract_hand(hand_landmarks):
        if hand_landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)

        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )

    def process_frame(self, frame_rgb):
        """Process one RGB frame and return 1662 features."""
        self.last_results = self.holistic.process(frame_rgb)
        results = self.last_results

        features = np.concatenate(
            [
                self._extract_face(results.face_landmarks).flatten(),
                self._extract_pose(results.pose_landmarks).flatten(),
                self._extract_hand(results.left_hand_landmarks).flatten(),
                self._extract_hand(results.right_hand_landmarks).flatten(),
            ]
        )

        return features.astype(np.float32)

    def draw_landmarks(self, frame_bgr):
        """Draw detected pose and hand landmarks on a BGR image."""
        if self.last_results is None:
            return frame_bgr

        results = self.last_results
        output = frame_bgr.copy()

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
            )

        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                output,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
            )

        return output

    def close(self):
        self.holistic.close()
