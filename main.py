import cv2
import os
import time
import datetime
import mediapipe as mp
import face_recognition
import pickle
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Create PoseDetector object for body detection
detector = PoseDetector()

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Output directory for videos
output_folder = "/Users/willweste/Door-Camera-Videos"

out = None


while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use MediaPipe Face Mesh for facial detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Use cvzone PoseDetector for body detection
    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=False)

    if (results is not None and results.multi_face_landmarks) or len(lmList) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            file_name = f"{current_time}.mp4"
            file_path = os.path.join(output_folder, file_name)
            out = cv2.VideoWriter(file_path, fourcc, 20, frame_size)
            print("Started Recording!")

    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        if detection:
            out.release()
        break

out.release()
cap.release()
cv2.destroyAllWindows()

