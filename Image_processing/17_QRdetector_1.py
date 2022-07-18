import numpy as np
import cv2 as cv
import qrcode


### Generate QRcode
generated_qr_code = qrcode.make("Learn Python")
generated_qr_code.save('Images/qrcode3.png')
generated_qr_code_cv = np.array(generated_qr_code)

### Detect QR code
qrcode_img = cv.imread("Images/qrcode3.png")

detector = cv.QRCodeDetector()
value, box, _ = detector.detectAndDecode(qrcode_img)
cv.rectangle(qrcode_img, (box[0][0][0], box[0][0][1]), (box[0][2][0], box[0][2][1]), (255,255,0), 2)


cv.imshow("qrcode_img", qrcode_img)
cv.waitKey()
cv.destroyAllWindows()

