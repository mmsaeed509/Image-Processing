#####################################
#                                   #
#  @author      : 00xWolf           #
#    GitHub    : @mmsaeed509       #
#    Developer : Mahmoud Mohamed   #
#  﫥  Copyright : Mahmoud Mohamed   #
#                                   #
#####################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.filters as filters
import scipy
import COLORS

# read the Image #
img = cv2.imread('CameraMan.png')

# convert the Image to gray level #
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Add Salt And Pepper Noise #
Salt_noise = skimage.util.random_noise(img_gray, mode='salt')
SaltAndPepperNoise = skimage.util.random_noise(Salt_noise, mode='pepper')

# Apply arithmetic mean filter #
mean = cv2.blur(SaltAndPepperNoise, (3, 3))

# Apply median filter #
median = filters.median(SaltAndPepperNoise)

# Apply max filter #
maximum_filter = scipy.ndimage.maximum_filter(SaltAndPepperNoise, 3)

# Apply min filter #
minimum_filter = scipy.ndimage.minimum_filter(SaltAndPepperNoise, 3)

# Writing the new Images #
cv2.imwrite("output-img/SaltAndPepperNoise.png", SaltAndPepperNoise)
cv2.imwrite("output-img/mean.png", mean)
cv2.imwrite("output-img/median.png", median)
cv2.imwrite("output-img/maximum_filter.png", maximum_filter)
cv2.imwrite("output-img/minimum_filter.png", minimum_filter)

# Plotting the images #
plt.subplot(1, 5, 1), plt.imshow(SaltAndPepperNoise, cmap='gray')
plt.title('Salt & Pepper'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 5, 2), plt.imshow(mean, cmap='gray')
plt.title('mean'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 5, 3), plt.imshow(median, cmap='gray')
plt.title('median'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 5, 3), plt.imshow(maximum_filter, cmap='gray')
plt.title('Max Filter'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 5, 3), plt.imshow(minimum_filter, cmap='gray')
plt.title('Min Filter'), plt.xticks([]), plt.yticks([])

plt.show()
