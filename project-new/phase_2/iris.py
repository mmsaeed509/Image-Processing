"""
# # # # # # # # # # # # # # # # # #
#                                 #
# 﫥  @author   : 00xWolf          #
#   GitHub    : @mmsaeed509      #
#   Developer : Mahmoud Mohamed  #
#                                 #
# # # # # # # # # # # # # # # # # #
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# ----------------- canny detection (edges detection) ----------------- #
img = cv2.imread('dataset/15.jpg')
edges = cv2.Canny(img, 100, 200)
cv2.imshow("edges", edges)
cv2.imwrite('edges.jpg', edges)
cv2.waitKey(1000)

# ----------------- Hough circle transform ( circle detection) ----------------- #

# arg 0 -> to read as a grey scale not RGB scale  #
img = cv2.imread('dataset/15.jpg', 0)
img = cv2.medianBlur(img, 5)  # improve result of circle detection #

# from grey scale to RGB #
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # cv2 always work with grey scale to BGR #
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))  # convert all numbers to integers not float #

# draw the detected circles #
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Iris region', cimg)
cv2.imwrite('circle.jpg', cimg)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ----------------- Thresholding ----------------- #

# arg 0 -> to read as a grey scale not RGB scale  #
img = cv2.imread('dataset/15.jpg', 0)
img = cv2.medianBlur(img, 5)  # improve result of circle detection #
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Binary Thresholding #
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Mean #
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Gaussian #
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
