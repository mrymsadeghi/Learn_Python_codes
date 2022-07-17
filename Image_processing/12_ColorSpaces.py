import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image_bgr = cv.imread("Images/blue_eyes.jpg")    #BGR
image_gr = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)

#blue_ch, green_ch, red_ch = cv.split(image)
image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
# RGB
# BGR  opencv
# BGRA / RGBA 4 channel alpha opacity

# HSV [20, 100, 100]
image_hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)


"""
cv.imshow('IMAGE display', image)
#cv.imshow('Blue display', blue_ch)
#cv.imshow('RED display', red_ch)
#cv.imshow('Green display', green_ch)
cv.waitKey(0)
cv.destroyAllWindows()
"""

plt.imshow(image_hsv)
plt.show()