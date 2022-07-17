
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

image = cv.imread("images/xray1.jpg", 0)
img_hist= cv.calcHist([image], [0],None,[256], [0,256])
equalized_histogram = cv.equalizeHist(image)
img_equal_hist= cv.calcHist([equalized_histogram], [0],None,[256], [0,256])

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)
cl_equal_hist= cv.calcHist([cl1], [0],None,[256], [0,256])


plt.subplot(231), plt.imshow(image, 'gray')
plt.subplot(234), plt.plot(img_hist)
plt.subplot(232), plt.imshow(equalized_histogram, 'gray')
plt.subplot(235), plt.plot(img_equal_hist)
plt.subplot(233), plt.imshow(cl1, 'gray')
plt.subplot(236), plt.plot(cl_equal_hist)
plt.show()