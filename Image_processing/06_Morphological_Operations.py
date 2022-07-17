

from cv2 import MORPH_OPEN
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


original_img = cv.imread("images/cells_threshold.png", 0)

_, mask = cv.threshold(original_img, 25, 255, cv.THRESH_BINARY)

# Erosion
kernel = np.ones((5,5), np.uint8)
#eroded_img = cv.erode(mask, kernel, iterations=1)

# Dilation
kernel2 = np.ones((25,25), np.uint8)
delated_img = cv.dilate(mask, kernel, iterations=1)
#kernel = np.ones((3,3), np.uint8)
#eroded_img = cv.erode(delated_img, kernel, iterations=2)
subtract_img = delated_img - mask

# Closing
#closed_img1 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel1)
#closed_img2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel2)

# Opening
#opened_img1 = cv.morphologyEx(mask, MORPH_OPEN, kernel1)
#opened_img2 = cv.morphologyEx(mask, MORPH_OPEN, kernel2)


# Gradient

gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel)












plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.show()

"""
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
"""