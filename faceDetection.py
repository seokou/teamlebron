import cv2
import mediapipe as mp
import sys

# Initialize Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,  # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices from MediaPipe
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    h, w, _ = frame.shape
    annotated = frame.copy()

    # Draw pose landmarks (body)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3)
        )
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x, y, z = lm.x, lm.y, lm.z
            print(f"[POSE] Landmark {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}", flush=True)

    # Draw face mesh and iris if visible
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
        )

        # Draw and print iris landmarks
        for idx in LEFT_IRIS + RIGHT_IRIS:
            lm = results.face_landmarks.landmark[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 2, (255, 0, 255), -1)
            print(f"[IRIS] Landmark {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}", flush=True)

    # Show the frame
    cv2.imshow("Holistic Tracker (Body + Face + Iris)", annotated)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
