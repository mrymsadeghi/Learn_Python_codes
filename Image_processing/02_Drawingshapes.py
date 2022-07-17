import numpy as np
import cv2 as cv

img = cv.imread("Images/cityscape.jpg", cv.IMREAD_GRAYSCALE)



cv.line(img, (5,5), (200,150), (200,200,0), 10)

cv.rectangle(img,(5,5), (200,150), (255,255,255), 8)

cv.circle(img, (200,150), 50, (0,255,255), 20)

points = np.array([[30,5], [300,200],[100,70],[40,100]], np.int32)
cv.polylines(img, [points], True, (255,0,0), 5)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img, 'Salam be hame',(20,20), font, 1, (255,255,255), 2, cv.LINE_AA)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()