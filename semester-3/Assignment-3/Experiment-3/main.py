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
from matplotlib import pyplot as plt
import COLORS

# read the Image #
print(COLORS.BOLD_PURPLE + "\n[*] Reading The Image" + COLORS.RESET_COLOR)
img = cv2.imread('CameraMan.png', 0)

# Add salt and pepper noise to the image #
print(COLORS.BOLD_PURPLE + "[+] Adding Salt And Pepper Noise" + COLORS.RESET_COLOR)

salt_vs_pepper = 0.5  # Salt to pepper ratio #
noise = np.zeros(img.shape, np.uint8)
cv2.randu(noise, 0, 255)
img_SaltAndPepper = img.copy()
img_SaltAndPepper[noise < 255 * salt_vs_pepper / 2] = 0
img_SaltAndPepper[noise > 255 * (1 - salt_vs_pepper / 2)] = 255

# Apply arithmetic mean filter #
print(COLORS.BOLD_PURPLE + "[+] Applying Arithmetic Mean Filter" + COLORS.RESET_COLOR)
img_mean = cv2.blur(img_SaltAndPepper, (3, 3))

# Apply median filter #
print(COLORS.BOLD_PURPLE + "[+] Applying Median Filter" + COLORS.RESET_COLOR)
img_median = cv2.medianBlur(img_SaltAndPepper, 3)

# Apply max filter #
print(COLORS.BOLD_PURPLE + "[+] Applying Max Filter" + COLORS.RESET_COLOR)

kernel = np.ones((3, 3), np.uint8)
img_max = cv2.dilate(img_SaltAndPepper, kernel)

# Apply min filter #
print(COLORS.BOLD_PURPLE + "[+] Applying Min Filter" + COLORS.RESET_COLOR)
img_min = cv2.erode(img_SaltAndPepper, kernel)

# Writing the new Images #
print(COLORS.BOLD_PURPLE + "[+] Writing The New Images" + COLORS.RESET_COLOR)

cv2.imwrite("output-img/SaltAndPepperNoise.png", img_SaltAndPepper)
cv2.imwrite("output-img/mean.png", img_mean)
cv2.imwrite("output-img/median.png", img_median)
cv2.imwrite("output-img/maximum_filter.png", img_max)
cv2.imwrite("output-img/minimum_filter.png", img_min)

# Plotting the images #
print(COLORS.BOLD_PURPLE + "[+] Plotting The Images" + COLORS.RESET_COLOR)

titles = ['Original', 'Salt & Pepper', 'Arithmetic Mean', 'Median', 'Max', 'Min']
images = [img, img_SaltAndPepper, img_mean, img_median, img_max, img_min]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

print(COLORS.BOLD_GREEN + "\n[✔] D O N E !" + COLORS.RESET_COLOR)
