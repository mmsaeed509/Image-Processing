# # # # # # # # # # # # # # # # # #
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
# # # # # # # # # # # # # # # # # #

# import libs #
import cv2
import numpy as np
import COLORS

# read image #
img = cv2.imread("kernel.png", cv2.IMREAD_COLOR)

# converted it to 2D Grey image #/home/ozil/.local/share/virtualenvs/Image-Processing-WgoHXI-i/bin/python /home/ozil/GitHub/FCAI-CU/Image-Processing/semester-3/Assignment-2/Task-1/main.py

grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# write new image #
cv2.imwrite("filter-imgs/kernel-grey.png", grey_image)


# Display the new grey image #
cv2.imshow("Grey Image", grey_image)
cv2.waitKey(0)

# ------------- Gaussian Noise ------------- #

gauss = np.random.normal(0, .5, grey_image.size)
gauss = gauss.reshape(grey_image.shape[0], grey_image.shape[1]).astype('uint8')

# Adding Gaussian Noise #
gaussian_img = cv2.add(grey_image, gauss)

# write new image #
cv2.imwrite("filter-imgs/kernel-grey-gaussian-noise.png", gaussian_img)

# Display the new Gaussian Noise Image #
cv2.imshow("Gaussian Noise Grey Image", gaussian_img)
cv2.waitKey(0)

# ------------- Smoothing Filters ------------- #

# ----- Average Filter ----- #

# read image #
average_image = cv2.imread("kernel.png", cv2.IMREAD_COLOR)
average_image = cv2.blur(average_image, (49, 49))

# write new image #
cv2.imwrite("filter-imgs/kernel-average-filter.png", average_image)

# Display the new Average Filter Image #
cv2.imshow("Average Filter Image", average_image)
cv2.waitKey(0)

# ----- Median Filter ----- #

# read image #
median_image = cv2.imread("kernel.png", cv2.IMREAD_COLOR)
# converted it to 2D Grey image #
grey_median_image = cv2.cvtColor(median_image, cv2.COLOR_BGR2GRAY)

median_image_filter = cv2.medianBlur(grey_median_image, 9)

# write new image #
cv2.imwrite("filter-imgs/kernel-median-filter.png", median_image_filter)

# Display the new Average Filter Image #
cv2.imshow("Median Filter Image", median_image_filter)
cv2.waitKey(0)

print(COLORS.PURPLE + "[" + COLORS.BOLD_GREEN + "✔" + COLORS.PURPLE +
      "] The output Images from filters is in " + COLORS.BOLD_CYAN +
      "filter-imgs" + COLORS.PURPLE + " directory \n" + COLORS.RESET_COLOR
      )
