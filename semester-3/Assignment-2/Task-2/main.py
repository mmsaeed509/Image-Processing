# # # # # # # # # # # # # # # # # #
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
# # # # # # # # # # # # # # # # # #

# import libs #
import cv2
import COLORS

# ------------- Sharpening Filters ------------- #

# ----- Sobel Filter ----- #

# read image #
sobel_image = cv2.imread("kernel.png", cv2.IMREAD_COLOR)

# converted it to 2D Grey image #
grey_sobel_image = cv2.cvtColor(sobel_image, cv2.COLOR_BGR2GRAY)

sobelX = cv2.Sobel(grey_sobel_image, cv2.CV_64F, 1, 0, ksize=3)
sobelY = cv2.Sobel(grey_sobel_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_image_filter = sobelX + sobelY

# Display Images #
cv2.imshow('Image sobel X filter', sobelX)
cv2.imshow('Image sobel y filter', sobelY)
cv2.imshow('Image sobel filter', sobel_image_filter)
cv2.waitKey(0)

# write new image #
cv2.imwrite("filter-imgs/kernel-sobel-x-filter.png", sobelX)
cv2.imwrite("filter-imgs/kernel-sobel-y-filter.jpg", sobelY)
cv2.imwrite("filter-imgs/kernel-sobel-filter.jpg", sobel_image_filter)

# ----- Laplacian Filter ----- #
laplacian_image = cv2.imread("kernel.png", cv2.IMREAD_COLOR)

# converted it to 2D Grey image #
grey_laplacian_image = cv2.cvtColor(laplacian_image, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter #
laplacian = cv2.Laplacian(grey_laplacian_image, cv2.CV_64F)

# Display Image #
cv2.imshow('Image Laplacian Filter', laplacian)
cv2.waitKey(0)

# write new image #
cv2.imwrite("filter-imgs/kernel-laplacian-filter.jpg", laplacian)

# Normalize the Laplacian-filtered image to 0-255 range #
laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display Image #
cv2.imshow('Image Normalize Laplacian Filter', laplacian_norm)
cv2.waitKey(0)

# write new image #
cv2.imwrite("filter-imgs/kernel-normalize-laplacian-filter.jpg", laplacian_norm)

# print Images directory #
print(COLORS.PURPLE + "[" + COLORS.BOLD_GREEN + "✔" + COLORS.PURPLE +
      "] The output Images from filters is in " + COLORS.BOLD_CYAN +
      "filter-imgs" + COLORS.PURPLE + " directory \n" + COLORS.RESET_COLOR
      )
