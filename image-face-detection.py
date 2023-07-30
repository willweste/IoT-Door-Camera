import cv2
import matplotlib.pyplot as plt

imagePath = 'assets/700-00190663en_Masterfile.jpg'

img = cv2.imread(imagePath)

shape = img.shape
# print("Dimensions of the array: " + str(shape))

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_shape = gray_image.shape
# print("Dimensions of the gray array: " + str(gray_shape))

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))
plt.imshow(rgb_image)
plt.axis('off')
plt.show()
