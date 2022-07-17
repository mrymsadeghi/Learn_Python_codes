import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv.VideoCapture(1)
detector = HandDetector(detectionCon = 0.5, maxHands=2)

while(True):
    rec, frame = cap.read()
    hand, frame = detector.findHands(frame)
    if hand:
        hand1 = hand[0]
        lmlist1 = hand1["lmList"]

        length, info, frame = detector.findDistance(lmlist1[4][:-1], lmlist1[8][:-1], frame)
        
    cv.imshow('frame', frame)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()