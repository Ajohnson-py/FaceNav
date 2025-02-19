import cv2 as cv
import mediapipe as mp

# Load face mesh from Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load image
webcam = cv.VideoCapture(1)

while True:
    ret, image = webcam.read()
    height, width, _ = image.shape
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Find facial landmarks
    result = face_mesh.process(rgb_image)

    for facial_landmark in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmark.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv.circle(image, (x, y), 4, (100, 100, 0), -1)

    cv.imshow("Image", image)
    cv.waitKey(1)
