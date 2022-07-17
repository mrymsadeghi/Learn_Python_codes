import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while(True):
    rec, frame = cap.read()
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red = np.array([100,50,50])
    upper_red = np.array([116,255,255])

    mask_red = cv.inRange(frame_hsv, lower_red, upper_red)
    frame_masked = cv.bitwise_and(frame, frame, mask = mask_red)

    cv.imshow('frame', frame)
    cv.imshow('mask_red', mask_red)
    cv.imshow('frame_masked', frame_masked)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()