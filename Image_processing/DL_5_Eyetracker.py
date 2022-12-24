import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv.VideoCapture(0)
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

while(True):
    rec, frame = cap.read()
    rec, frame2 = cap.read()
    frame, bbox = detector.findFaces(frame)
    frame, faces = meshdetector.findFaceMesh(frame)

    if bbox:
        center = bbox[0]["center"]   
        i=15
        if faces:
            for i in range(100, len(faces[0])):
                cv.putText(frame2, str(i), (faces[0][i][0],faces[0][i][1]), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv.LINE_AA)

        #cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)
    cv.imshow('frame', frame2)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()