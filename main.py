import os
import cv2
import time
import datetime

import numpy as np
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)

# Load the DNN model for face detection
model_path = "deploy.prototxt"
weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

# Create PoseDetector object for body detection
detector = PoseDetector()

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

# Output directory for videos
output_folder = "/Users/willweste/Door-Camera-Videos"

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the DNN model for face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Check for faces in the detections
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Minimum confidence threshold for face detection
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    # Use cvzone PoseDetector for body detection
    img = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=False)

    if len(faces) + len(lmList) > 0:
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

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)

    # Draw pose keypoints on the frame
    if lmList:
        center = (int(lmList[14][1]), int(lmList[14][2]))
        cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        if detection:
            out.release()
        break

out.release()
cap.release()
cv2.destroyAllWindows()
