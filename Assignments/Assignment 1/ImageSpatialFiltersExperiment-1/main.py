"""
Names :-                                IDs :-
        Mahmoud Mohamed Said Ahmed             20180261
        Abdallah Adham Sharkawy                20180161
        Hassan Khamis                          20180087

"""

import cv2
import numpy as np

from matplotlib import pyplot as plt


# Histogram

# reads an input image
img = cv2.imread("luftwaffe.png", cv2.IMREAD_COLOR)
# converted it to 2D Grey img
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plotting histogram for the original 2D Grey img
cv2.imwrite("djfsd.png",image)

#plt.hist(image)
#plt.title("Image Histogram")
plt.xlabel("greyscale")
plt.ylabel("freq")
# plt.figure()
# plt.show()

# Histogram

# Histogram equalization
HisEqu = cv2.equalizeHist(image)
cv2.imshow("Image Equalization", HisEqu)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# writing img to file
plt.hist(HisEqu)
plt.title("Equalized Image Histogram ")
plt.xlabel("grayscale value")
plt.ylabel("freq")
plt.figure()
plt.show()

# Histogram equalization

# smoothing filters

# Average filter
blurred = cv2.blur(image, (10, 10))
cv2.imshow("Image average filter", blurred)
cv2.waitKey(4000)
cv2.destroyAllWindows()


# Median filter
medBlur = cv2.medianBlur(image, 5)
cv2.imshow('Image median filter', medBlur)
cv2.waitKey(4000)
cv2.destroyAllWindows()


# Sobel filter

sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
print("Sobel X filter applied")
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
print("Sobel Y filter applied")
cv2.imshow('Image sobel X filter', sobelX)
cv2.imshow('Image sobel y filter', sobelY)
cv2.waitKey(4000)
cv2.destroyAllWindows()


# Sobel filter



# smoothing filters
