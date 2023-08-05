import cv2
import os
import time
import datetime
import mediapipe as mp
from app_ntfy import notify_app

from video_wiper import clean_up_old_videos
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=3, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Create PoseDetector object for body detection
detector = PoseDetector()

# Start Recording Parameters
detection = False
detection_start_time = None
SECONDS_TO_START_RECORDING = 5

# Stop Recording Parameters
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 8

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Output directory for videos
output_folder = "/Users/willweste/Door-Camera-Videos"

out = None

try:
    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use MediaPipe Face Mesh for facial detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        # Use MediaPipe Pose for body detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)

        # Use cvzone PoseDetector for body detection
        frame = detector.findPose(frame)
        lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=False)

        if results.multi_face_landmarks or results_pose.pose_landmarks or len(lmList) > 0:
            if not detection:
                detection_start_time = time.time()
                detection = True

            if time.time() - detection_start_time >= SECONDS_TO_START_RECORDING:
                if out is None:
                    # Send notification to phones
                    notify_app()
                    current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                    file_name = f"{current_time}.mp4"
                    file_path = os.path.join(output_folder, file_name)
                    out = cv2.VideoWriter(file_path, fourcc, 20, frame_size)
                    print("Started Recording!")
                    clean_up_old_videos()
        else:
            if detection:
                if timer_started:
                    if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                        detection = False
                        timer_started = False
                        out.release()
                        out = None
                        print('Stop Recording!')
                else:
                    timer_started = True
                    detection_stopped_time = time.time()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh landmarks on the frame
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, c = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Draw the body landmarks on the frame
        if results_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if out is not None:
            out.write(frame)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    if out is not None:
        out.release()

cap.release()
cv2.destroyAllWindows()
