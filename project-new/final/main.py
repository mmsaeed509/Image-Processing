"""
######################################
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
######################################
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# ----------------- canny detection (edges detection) ----------------- #
IMAGE = 'dataset/15.jpg'
IMAGE_2 = 'dataset/13.png'
img = cv2.imread(IMAGE_2)
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Edges", edges)
cv2.imwrite('edges.jpg', edges)
cv2.waitKey(1000)

# ----------------- Hough circle transform ( circle detection ) ----------------- #

# arg 2 -> to read as a grey scale not RGB scale  #
img = cv2.imread(IMAGE_2, 0)
img = cv2.medianBlur(img, 5)  # improve result of circle detection #

# from grey scale to RGB #
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # cv2 always work with grey scale to BGR #

# arg 1 -> image. , arg 2 -> detection method (HOUGH_GRADIENT).
# arg 3 -> dp : inverse ratio of the accumulator resolution to the image resolution.
# arg 4 -> minDist : minimum distance between the centers of the detected circles.
# arg 5 -> the higher threshold of the two passed to the Canny edge detector.
# arg 6 -> the accumulator threshold for the circle centers at the detection stage.
# arg 7 -> minimum circle radius.
# arg 8 -> maximum circle radius.
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))  # convert all numbers to integers not float #

# draw the detected circles #
for i in circles[0, :]:
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Circle Detection', cimg)
cv2.imwrite('circle.jpg', cimg)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# ----------------- Thresholding ----------------- #

# arg 2 -> to read as a grey scale not RGB scale  #
img = cv2.imread(IMAGE_2, 0)
img = cv2.medianBlur(img, 5)  # improve result of circle detection #
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Binary Thresholding #
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Mean #
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)  # Adaptive Mean #
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# ----------------- Segmentation (Final) ----------------- #

# Read in the image 3
image = cv2.imread(IMAGE_2)

# Change color to RGB (from BGR) #
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB) #
# -1 for auto complete the new matrix shape #
pixel_vals = image.reshape((-1, 3))

# Convert to float type #
pixel_vals = np.float32(pixel_vals)

# the below line of code defines the criteria for the algorithm to stop running (to stop algo)
# which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
# becomes 85%                                              No. iterations   required accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

# then perform k-means clustering wit h number of clusters defined as 3
# also random centres are initially chose for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

# reshape data into the original image dimensions
segmented_image = segmented_data.reshape(image.shape)
print(set(labels.flatten()))
cv2.imshow("Oil", segmented_image)
cv2.imwrite('segmented_image.jpg', segmented_image)
cv2.waitKey(1000000)
