import numpy as np
import cv2 as cv

def rescale_frame(frame, percent=20):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation =cv.INTER_AREA)


cap = cv.VideoCapture("Videos/street_camera.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)
while(True):
    rec, frame1 = cap.read()
    rec, frame2 = cap.read()

    resized_frame1 = rescale_frame(frame1)
    resized_frame2 = rescale_frame(frame2)

    frame_diff = cv.absdiff(resized_frame1, resized_frame2)
    frame_diff_gr = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(frame_diff_gr, (9,9), 1)
    _, mask = cv.threshold(blurred_frame, 10, 255, cv.THRESH_BINARY)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    for contour in contours:
        if cv.contourArea(contour)>1000:
            (x, y, w, h) = cv.boundingRect(contour)
            cv.rectangle(resized_frame1, (x + w -100, y + h-50), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('frame', resized_frame1)
    cv.imshow('frame_diff', frame_diff)
    cv.imshow('mask', mask)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()