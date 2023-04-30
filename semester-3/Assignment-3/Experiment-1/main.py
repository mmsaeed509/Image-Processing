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
import COLORS

# read the Image #
print(COLORS.BOLD_PURPLE + "\n[*] Reading The Image" + COLORS.RESET_COLOR)
img = cv2.imread('CameraMan.png')

# convert the Image to gray level #
print(COLORS.BOLD_PURPLE + "[+] converting the Image to gray level" + COLORS.RESET_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Creating Low/high pass filters using a kernel that enhances the edges of the image. #
# Creating a 5x5 low pass filter #
print(COLORS.BOLD_PURPLE + "[+] Applying Low Pass Filter" + COLORS.RESET_COLOR)

low_pass_filter_img_kernel = np.ones((5, 5), np.float32) / 25
# using `filter2D()` method to apply the filter on the grey image #
low_pass_filter_img = cv2.filter2D(img_gray, -1, low_pass_filter_img_kernel)

# Creating a 3x3 high pass filter #
print(COLORS.BOLD_PURPLE + "[+] Applying High Pass Filter" + COLORS.RESET_COLOR)

high_pass_filter_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
high_pass_filter_img = cv2.filter2D(img_gray, -1, high_pass_filter_kernel)

# Writing the new Images #
print(COLORS.BOLD_PURPLE + "[+] Writing The New Images" + COLORS.RESET_COLOR)

cv2.imwrite("output-img/low_pass_filter_img.png", low_pass_filter_img)
cv2.imwrite("output-img/high_pass_filter_img.png", high_pass_filter_img)

# Plotting the images #
print(COLORS.BOLD_PURPLE + "[+] Plotting The Images" + COLORS.RESET_COLOR)

plt.subplot(1, 3, 1), plt.imshow(img_gray, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(low_pass_filter_img, cmap='gray')
plt.title('Low Pass Filtered'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.imshow(high_pass_filter_img, cmap='gray')
plt.title('High Pass Filtered'), plt.xticks([]), plt.yticks([])

plt.show()

print(COLORS.BOLD_GREEN + "\n[✔] D O N E !" + COLORS.RESET_COLOR)
