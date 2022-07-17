import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread("Images/cityscape.jpg", cv.IMREAD_GRAYSCALE)

cv.imwrite("Images/cityscape2.jpg", image)
#cv.IMREAD_GRAYSCALE    0
#cv.IMREAD_COLOR        1
#cv.IMREAD_UNCHANGED    -1


"""
cv.imshow('IMAGE display', image)
cv.waitKey(0)
cv.destroyAllWindows()
"""

plt.imshow(image)
plt.imshow()