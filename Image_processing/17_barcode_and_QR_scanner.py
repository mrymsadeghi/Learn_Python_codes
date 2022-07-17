
import numpy as np
import cv2 as cv
from pyzbar.pyzbar import decode

barcode_img = cv.imread("Images/barcode1.jpg", 0)
decoded_barcode = decode(barcode_img)
print(decoded_barcode)