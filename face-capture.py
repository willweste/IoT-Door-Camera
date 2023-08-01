import cv2
import dlib

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to capture your face and save its landmarks
def capture_reference_face():
    cap = cv2.VideoCapture(0)  # Access the computer's webcam (use 1 if external webcam)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = detector(gray)

        if len(faces) == 1:  # Assuming only one face is present
            # Predict facial landmarks
            landmarks = predictor(gray, faces[0])

            # Save landmarks to a file
            with open('reference_landmarks.txt', 'w') as file:
                for n in range(0, 68):
                    x_lm, y_lm = landmarks.part(n).x, landmarks.part(n).y
                    file.write(f"{x_lm} {y_lm}\n")
            break

        cv2.imshow('Capture Reference Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to capture your face and save its landmarks
capture_reference_face()
