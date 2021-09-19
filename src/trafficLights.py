__author__ = 'Tiago Tamagusko <tamagusko@gmail.com>'
__version__ = 'dev 0.1 (2021-04-22)'
__license__ = 'Proprietary'

# Source 1: https://github.com/HevLfreis/TrafficLight-Detector
# Source 2: https://stackoverflow.com/questions/58109962/how-to-optimize-circle-detection-with-python-opencv

import os
import cv2
import numpy as np


def detect(filepath, file):

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(filepath+file)
    _, width = img.shape[:2]
    maxRadius = int(1.1*(width/12)/2)
    minRadius = int(0.5*(width/12)/2)
    cimg = img
    # convert img to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    # red mask 1 (0-10)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    # red mask 1  (160-180)
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15, 150, 150])
    upper_yellow = np.array([35, 255, 255])
    maskr1 = cv2.inRange(hsv, lower_red1, upper_red1)
    maskr2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(maskr1, maskr2)
    # create more mask values?

    size = img.shape
    # print size

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2*minRadius,
                                 param1=50, param2=4, minRadius=0, maxRadius=maxRadius)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2*minRadius,
                                 param1=50, param2=10, minRadius=0, maxRadius=maxRadius)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, dp=1.2, minDist=2*minRadius,
                                 param1=50, param2=5, minRadius=0, maxRadius=maxRadius)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 0, 255), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                # cv2.putText(cimg, 'RED', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                # cv2.putText(cimg, 'GREEN',  (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 255), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                # cv2.putText(cimg, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # transform 2d to 3d
    maskr_rgb = np.stack([maskr]*3, axis=2)
    maskg_rgb = np.stack([maskg]*3, axis=2)
    masky_rgb = np.stack([masky]*3, axis=2)
    # sum images and masks
    images1 = np.hstack((cimg, maskr_rgb))
    images2 = np.hstack((maskg_rgb, masky_rgb))
    images = np.vstack((images1, images2))

    # cv2.imshow('detected results', cimg)
    cv2.imshow('processed - red / green - yellow', images)
    # cv2.imshow('mask red', maskr)
    # cv2.imshow('mask green', maskg)
    # cv2.imshow('mask yellow', maskg)
    cv2.imwrite(results+file, cimg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = '/home/t1/Dropbox/09-Jest/09-MyWorkshops/workshop-intro-opencv/img/trafficLights/'
results = '/home/t1/Dropbox/09-Jest/09-MyWorkshops/workshop-intro-opencv/output/'

for f in os.listdir(path):
    print(f)
    if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpeg'):
        detect(path, f)
