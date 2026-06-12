import mediapipe as mp
import cv2
import json
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
print("Pose server baslatildi! Cikis icin ESC")

OUTPUT_FILE = "pose_data.json"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        data = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            data[str(i)] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'v': lm.visibility
            }
        try:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(data, f)
        except:
            pass

        # Draw landmarks on frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Pose server durduruldu.")
