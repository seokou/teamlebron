import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe tools
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# Set up FaceMesh (max 5 faces)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set up Selfie Segmentation for detecting bodies
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # No min_detection_confidence

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Convert frame to RGB (MediaPipe works with RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(rgb_frame)
    results_segmentation = selfie_segmentation.process(rgb_frame)

    # Get image dimensions
    h, w, _ = frame.shape

    # Prepare output frame
    annotated_frame = frame.copy()

    # Face Mesh: Draw landmarks and count faces
    if results_face.multi_face_landmarks:
        face_count = len(results_face.multi_face_landmarks)
        for face_idx, face_landmarks in enumerate(results_face.multi_face_landmarks):
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Print x, y, z coordinates for each landmark
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z  # Depth (distance from camera)
                print(f"Face {face_idx+1} - Landmark Coordinates (x, y, z): ({x}, {y}, {z})")
        
        print(f"[INFO] Faces detected: {face_count}")
    else:
        face_count = 0
        print("[INFO] No faces detected")

    # Selfie Segmentation: Count bodies detected using segmentation mask
    if results_segmentation.segmentation_mask is not None:
        mask = results_segmentation.segmentation_mask
        mask = np.uint8(mask * 255)  # Convert to a binary mask (0 or 255)

        # Apply threshold to create a binary mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours to count separate objects (people)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        body_count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 500])  # Filter small contours

        # Draw segmentation mask for visualization
        annotated_frame = cv2.addWeighted(annotated_frame, 1, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.5, 0)

        print(f"[INFO] Estimated people in frame: {body_count}")

    else:
        body_count = 0
        print("[INFO] No body detected")

    # Compare face detection vs body detection (discrepancy check)
    if body_count != face_count:
        print(f"[WARNING] Discrepancy detected! Faces: {face_count}, Bodies: {body_count}")

    # Display annotated frame
    cv2.imshow("Face + Body Detection (Discrepancy Check)", annotated_frame)

    # Exit on 'Esc' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
