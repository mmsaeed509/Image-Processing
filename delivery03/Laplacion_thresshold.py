import cv2
import numpy as np

img = cv2.imread('cameramanN1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
image = cv2.convertScaleAbs(image)
cv2.imshow("Laplacian image", image)

img = cv2.resize(img, (500, 400))
retVal, dst = cv2.threshold(img, 120, 200, cv2.THRESH_BINARY)
cv2.imshow("thresholded image", dst)
cv2.waitKey()
