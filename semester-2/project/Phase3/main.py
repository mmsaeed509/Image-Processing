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
import matplotlib.pyplot as plt
import cv2

# Read in the image 3
image = cv2.imread('dataset/13.png')

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
cv2.imshow("Segmented image", segmented_image)
cv2.waitKey(1000000)
