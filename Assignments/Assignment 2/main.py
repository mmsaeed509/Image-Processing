import cv2
import numpy as np
import scipy
import skimage
import skimage.filters as imgFilters
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener

img = cv2.imread("cameramanN1.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gauss = np.random.normal(0, .5, img.size)
gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')

# Add the Gaussian noise to the image
img_gauss = cv2.add(img, gauss)
# Display the image
cv2.imshow('gauss_noise', img_gauss)
# cv2.waitKey(0)


WienerFilteredImage = wiener(img_gauss, None, 1)  # Filter the image
WienerFilteredImage = WienerFilteredImage.reshape(img_gauss.shape[0], img_gauss.shape[1]).astype('uint8')
cv2.imshow('Wiener_Filter', WienerFilteredImage)

Salt_noise = skimage.util.random_noise(img, mode='salt')
SaltAndPepperNoise = skimage.util.random_noise(Salt_noise, mode='pepper')
cv2.imshow('s&p', SaltAndPepperNoise)

mean = cv2.blur(SaltAndPepperNoise,(3,3))
cv2.imshow('mean filter', mean)

median = imgFilters.median(SaltAndPepperNoise)
cv2.imshow('median filter', median)


minimum_filter = scipy.ndimage.minimum_filter(SaltAndPepperNoise, 3)
cv2.imshow('minimum filter', minimum_filter)

maximum_filter = scipy.ndimage.maximum_filter(SaltAndPepperNoise,3)
cv2.imshow('maximum filter', maximum_filter)

cv2.waitKey(0)
