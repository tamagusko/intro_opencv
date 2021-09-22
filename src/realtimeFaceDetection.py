__author__ = 'Tiago Tamagusko <tamagusko@gmail.com>'
__version__ = 'dev 0.1 (2021-09-22)'
__license__ = 'Proprietary'

import cv2

# define a video capture channel
cap = cv2.VideoCapture(0)

# cascade face classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:  # infinite loop
    _, frame = cap.read()  # is runing, frame captured
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_frame,  # input image
        scaleFactor=1.1,
        minNeighbors=3
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF  # exit with ESC key
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
