#####################################
#                                   #
#  @author      : 00xWolf           #
#    GitHub    : @mmsaeed509       #
#    Developer : Mahmoud Mohamed   #
#  﫥  Copyright : Mahmoud Mohamed   #
#                                   #
#####################################

import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
import numpy as np
import COLORS

# read the Image #
print(COLORS.BOLD_PURPLE + "\n[*] Reading The Image" + COLORS.RESET_COLOR)
img = cv2.imread('eight.tif')

# convert the Image to gray level #
print(COLORS.BOLD_PURPLE + "[+] converting the Image to gray level" + COLORS.RESET_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adding Gaussian noise to the image #
print(COLORS.BOLD_PURPLE + "[+] Adding Gaussian Noise" + COLORS.RESET_COLOR)

noise = np.zeros(img_gray.shape, dtype=np.uint8)
cv2.randn(noise, 0, 25)  # mean = 0, standard deviation = 25
img_gauss = cv2.add(img_gray, noise)

# Applying Adaptive Wiener filter to the noisy image #
print(COLORS.BOLD_PURPLE + "[+] Applying Adaptive Wiener Filter" + COLORS.RESET_COLOR)

WienerFilteredImage = wiener(img_gauss, None, 1)  # Filter the image
WienerFilteredImage = WienerFilteredImage.reshape(img_gauss.shape[0], img_gauss.shape[1]).astype('uint8')

# Writing the new Images #
print(COLORS.BOLD_PURPLE + "[+] Writing The New Images" + COLORS.RESET_COLOR)

cv2.imwrite("output-img/img_gauss_eight.png", img_gauss)
cv2.imwrite("output-img/WienerFilteredImage_eight.png", WienerFilteredImage)

# Plotting the images #

print(COLORS.BOLD_PURPLE + "[+] Plotting The Images" + COLORS.RESET_COLOR)

plt.subplot(1, 2, 1), plt.imshow(img_gauss, cmap='gray')
plt.title('eight Gaussian Noise Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(WienerFilteredImage, cmap='gray')
plt.title('eight Wiener Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()

print(COLORS.BOLD_GREEN + "\n[✔] D O N E !" + COLORS.RESET_COLOR)

# Apply on CameraMan #

# read the Image #
print(COLORS.BOLD_PURPLE + "\n[*] Reading The Image" + COLORS.RESET_COLOR)
img = cv2.imread('CameraMan.png')

# convert the Image to gray level #
print(COLORS.BOLD_PURPLE + "[+] converting the Image to gray level" + COLORS.RESET_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adding Gaussian noise to the image #
print(COLORS.BOLD_PURPLE + "[+] Adding Gaussian Noise" + COLORS.RESET_COLOR)

noise = np.zeros(img_gray.shape, dtype=np.uint8)
cv2.randn(noise, 5, 25)  # mean = 5 (to avoid divide-by-zero ), standard deviation = 25
img_gauss = cv2.add(img_gray, noise)


# Applying Adaptive Wiener filter to the noisy image #
print(COLORS.BOLD_PURPLE + "[+] Applying Adaptive Wiener Filter" + COLORS.RESET_COLOR)

WienerFilteredImage = wiener(img_gauss, None, 1)  # Filter the image
WienerFilteredImage = WienerFilteredImage.reshape(img_gauss.shape[0], img_gauss.shape[1]).astype('uint8')

# Writing the new Images #
print(COLORS.BOLD_PURPLE + "[+] Writing The New Images" + COLORS.RESET_COLOR)

cv2.imwrite("output-img/img_gauss_CameraMan.png", img_gauss)
cv2.imwrite("output-img/WienerFilteredImage_CameraMan.png", WienerFilteredImage)

# Plotting the images #
print(COLORS.BOLD_PURPLE + "[+] Plotting The Images" + COLORS.RESET_COLOR)

plt.subplot(1, 2, 1), plt.imshow(img_gauss, cmap='gray')
plt.title('CameraMan Gaussian Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.imshow(WienerFilteredImage, cmap='gray')
plt.title('CameraMan Wiener Filtered'), plt.xticks([]), plt.yticks([])

plt.show()

print(COLORS.BOLD_GREEN + "\n[✔] D O N E !" + COLORS.RESET_COLOR)
