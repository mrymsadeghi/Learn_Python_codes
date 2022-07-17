
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# Read in the image
image = cv.imread('Images/shapes.png')
image_gr = cv.imread('Images/shapes.png', 0)

corners = cv.cornerHarris(image_gr, 3, 10, 0.04)
corners_dilated = cv.dilate(corners, None)

image[corners_dilated > 0.01*corners_dilated.max()] = [255,0,0]


plt.imshow(image)
plt.imshow()