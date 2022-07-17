import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv.VideoCapture(1)
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

while(True):
    rec, frame = cap.read()
    frame, bbox = detector.findFaces(frame)
    frame, faces = meshdetector.findFaceMesh(frame)

    if bbox:
        center = bbox[0]["center"]   
        #cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)
    cv.imshow('frame', frame)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()