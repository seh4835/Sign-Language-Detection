import cv2
import mediapipe as mp


VIDEO_PATH = "data/raw/msasl/again__train__nvaS2J5mcEk.mp4"


mp_holistic = mp.solutions.holistic


holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open test video.")
    raise SystemExit(1)


fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 70)
print("MEDIAPIPE HOLISTIC TEST")
print("=" * 70)

print(f"Video FPS: {fps:.2f}")

success, frame = cap.read()

if not success:
    print("ERROR: Could not read first frame.")
    cap.release()
    holistic.close()
    raise SystemExit(1)


rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

results = holistic.process(rgb)


print()
print("Face detected:", results.face_landmarks is not None)
print("Pose detected:", results.pose_landmarks is not None)
print("Left hand detected:", results.left_hand_landmarks is not None)
print("Right hand detected:", results.right_hand_landmarks is not None)


if results.face_landmarks:
    print(
        "Face landmarks:",
        len(results.face_landmarks.landmark)
    )

if results.pose_landmarks:
    print(
        "Pose landmarks:",
        len(results.pose_landmarks.landmark)
    )

if results.left_hand_landmarks:
    print(
        "Left-hand landmarks:",
        len(results.left_hand_landmarks.landmark)
    )

if results.right_hand_landmarks:
    print(
        "Right-hand landmarks:",
        len(results.right_hand_landmarks.landmark)
    )


cap.release()
holistic.close()

print()
print("Holistic test finished.")