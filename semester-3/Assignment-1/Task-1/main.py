# # # # # # # # # # # # # # # # # #
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
# # # # # # # # # # # # # # # # # #

import cv2
import numpy as np
from matplotlib import pyplot as plt

# reads an input image #
img = cv2.imread("luftwaffe.png", cv2.IMREAD_COLOR)

# converted it to 2D Grey img #
grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plotting histogram for the original 2D Grey img #
cv2.imwrite("luftwaffe-grey.png", grey_image)  # write new img #


# calculate Cumulative Histogram #
def calculate_cumulative(histogram_input):
    cumulative_histogram_list = []
    temp = 0
    for i in range(0, 255):
        temp = temp + histogram_input[i, 0]
        cumulative_histogram_list.append(temp)

    return cumulative_histogram_list


# calculate the histogram #
"""
arg1 -> img, arg2 -> channels, arg3 -> mask (8 bit array) same size like input image.
arg4 -> histogram sizes in each dimension, arg5 -> ranges
"""
histogram = cv2.calcHist([grey_image], [0], None, [256], [0, 255])

# show plotting #
plt.plot(histogram, color='g')
plt.title("Image Histogram")
plt.xlabel("greyscale")
plt.ylabel("freq")
plt.figure()
plt.show()

# plt.hist(grey_image)
# plt.title("Image Histogram 2 ")
# plt.xlabel("greyscale")
# plt.ylabel("freq")
# plt.figure()
# plt.show()
#
# plt.hist(grey_image.ravel(),256,[0,256])
# plt.title('Image Histogram 3')
# plt.xlabel("greyscale")
# plt.ylabel("freq")
# plt.figure()
# plt.show()

# Histogram shift by value #
shifted_Histogram = cv2.calcHist([grey_image], [0], None, [256], [100, 255])
plt.plot(shifted_Histogram, color='r')
plt.title('Shifted Image Histogram')
cv2.imwrite("shifted_Histogram.png", shifted_Histogram)  # write new img #
plt.show()

# Cumulative Image Histogram #
plt.plot(calculate_cumulative(histogram), color='b')
plt.title("Cumulative Image Histogram")
plt.figure()
plt.show()

# Histogram equalization #
histogram_equalization = cv2.equalizeHist(grey_image)
cv2.imshow("Image Equalization", histogram_equalization)
cv2.waitKey(1000)
# cv2.destroyAllWindows()
cv2.imwrite("luftwaffe-grey-histogram-equalization.png", histogram_equalization)  # write new img #

plt.hist(histogram_equalization)
plt.title("Equalized Image Histogram ")
plt.xlabel("grayscale value")
plt.ylabel("freq")
plt.figure()
plt.show()

cumulative_histogram_equalization = calculate_cumulative(histogram_equalization)
plt.hist(cumulative_histogram_equalization)
plt.title("Equalized Image cumulative Histogram ")
plt.xlabel("grayscale value")
plt.ylabel("freq")
plt.figure()
plt.show()
