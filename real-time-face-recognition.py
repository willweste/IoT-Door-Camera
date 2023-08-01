import cv2
import dlib
import numpy as np

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the reference facial landmarks from the saved file
reference_landmarks = []
with open('reference_landmarks.txt', 'r') as file:
    for line in file:
        x, y = map(int, line.strip().split())
        reference_landmarks.append((x, y))

# Function to compute Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to recognize your face in real-time
def recognize_face_real_time():
    cap = cv2.VideoCapture(0)  # Access the computer's webcam (use 1 if external webcam)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        for face in faces:
            # Predict facial landmarks
            landmarks = predictor(gray, face)

            # Convert facial landmarks to a list of (x, y) coordinates
            detected_landmarks = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(0, 68)]

            # Compute the mean Euclidean distance between detected landmarks and reference landmarks
            mean_distance = np.mean([euclidean_distance(detected_landmarks[i], reference_landmarks[i]) for i in range(68)])

            # Define a threshold to determine if it's your face or not
            threshold = 10  # You may adjust this threshold based on your reference face

            # Recognize your face based on the threshold
            if mean_distance < threshold:
                # If the mean distance is below the threshold, it's your face
                name = "Your Name"  # Replace "Your Name" with your actual name
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # If the mean distance is above the threshold, it's not your face
                cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw a rectangle around the face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Visualize facial landmarks
            for n in range(0, 68):
                x_lm, y_lm = detected_landmarks[n]
                cv2.circle(frame, (x_lm, y_lm), 2, (255, 0, 0), -1)

        cv2.imshow('Real-Time Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function for real-time face recognition
recognize_face_real_time()
