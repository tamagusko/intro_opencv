__author__ = 'Tiago Tamagusko <tamagusko@gmail.com>'
__version__ = 'dev 0.1 (2021-09-22)'
__license__ = 'Proprietary'

import cv2

# define a video capture channel
cap = cv2.VideoCapture(0)

while True:  # infinite loop
    _, frame = cap.read()  # is runing, frame captured
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF  # exit with ESC key
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
