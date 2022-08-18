import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 

barcode_img = cv.imread("Images/qrcode1.png")


def QR_decode(im) : 
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)
    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data,'/n')     
    return decodedObjects

# Display barcode and QR code location  
def display(im, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects: 
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4 : 
          hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))
        else : 
          hull = points
        # Number of points in the convex hull
        n = len(hull)
        # Draw the convext hull
        for j in range(0,n):
          cv.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
    return im


cap = cv.VideoCapture(1)
while(True):
    rec, frame = cap.read()
    frame_gr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    decodedObjects = QR_decode(frame_gr)
    for decodedObject in decodedObjects: 
        points = decodedObject.polygon
     
        # If the points do not form a quad, find convex hull
        if len(points) > 4 : 
          hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
          hull = list(map(tuple, np.squeeze(hull)))
        else : 
          hull = points;
         
        # Number of points in the convex hull
        n = len(hull)     
        # Draw the convext hull
        for j in range(0,n):
          cv.line(frame, hull[j], hull[ (j+1) % n], (255,0,0), 3)

        x = decodedObject.rect.left
        y = decodedObject.rect.top

        print(x, y)

        print('Type : ', decodedObject.type)
        print('Data : ', decodedObject.data,'/n')

        barCode = str(decodedObject.data)
        cv.putText(frame, barCode, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv.LINE_AA) 
    cv.imshow('frame', frame)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()