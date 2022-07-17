
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


image = cv.imread('images/Road_in_Norway.jpg', 0)
image_noise_removed = cv.GaussianBlur(image, (3,3),0)

# Laplacian
laplacian = cv.Laplacian(image_noise_removed, cv.CV_64F)

# sobelx
sobelx= cv.Sobel(image_noise_removed, cv.CV_64F, 1, 0, ksize=5)

# sobely
sobely= cv.Sobel(image_noise_removed, cv.CV_64F, 0, 1, ksize=5)


# Canny Edge Detection
canny = cv.Canny(image_noise_removed, 100, 200)


















"""
plt.subplot(1,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

plt.show()

"""
plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(canny,cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
