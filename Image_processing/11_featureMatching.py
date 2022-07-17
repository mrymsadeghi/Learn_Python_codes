import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


image1 = cv.imread("images/book1.png")
image2 = cv.imread("images/book2.png")

#ORB
feat_orb = cv.ORB_create(nfeatures=1000)


orb_keypoints1, descriptors1 = feat_orb.detectAndCompute(image1, None)
orb_keypoints2, descriptors2 = feat_orb.detectAndCompute(image2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= True)

matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key= lambda x:x.distance)

image_matches = cv.drawMatches(image1, orb_keypoints1, image2, orb_keypoints2, matches[:100], None, flags=2)

plt.imshow(image_matches)
plt.show()
