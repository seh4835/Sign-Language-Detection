import numpy as np
import mediapipe as mp


class HolisticExtractor:
    """
    Extracts MediaPipe Holistic landmarks from a video frame.

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

    FEATURE_DIM = (
        FACE_LANDMARKS * 3
        + POSE_LANDMARKS * 4
        + HAND_LANDMARKS * 3
        + HAND_LANDMARKS * 3
    )

    def __init__(
        self,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_holistic = mp.solutions.holistic

        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _extract_face(face_landmarks):
        if face_landmarks is None:
            return np.zeros((468, 3), dtype=np.float32)

        return np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in face_landmarks.landmark
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _extract_pose(pose_landmarks):
        if pose_landmarks is None:
            return np.zeros((33, 4), dtype=np.float32)

        return np.array(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    landmark.visibility,
                ]
                for landmark in pose_landmarks.landmark
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _extract_hand(hand_landmarks):
        if hand_landmarks is None:
            return np.zeros((21, 3), dtype=np.float32)

        return np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

    def process_frame(self, frame_rgb):
        """
        Process one RGB frame and return a 1662-dimensional vector.
        """

        results = self.holistic.process(frame_rgb)

        face = self._extract_face(results.face_landmarks)
        pose = self._extract_pose(results.pose_landmarks)
        left_hand = self._extract_hand(results.left_hand_landmarks)
        right_hand = self._extract_hand(results.right_hand_landmarks)

        features = np.concatenate(
            [
                face.flatten(),
                pose.flatten(),
                left_hand.flatten(),
                right_hand.flatten(),
            ]
        )

        return features.astype(np.float32)

    def close(self):
        self.holistic.close()