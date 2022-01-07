import cv2
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal.signaltools import wiener
from skimage.util import random_noise
from scipy import ndimage
import numpy as np
#----------#
img = cv.imread("C:\\Users\Hashem\Desktop\eye2.jpg",0)

#------------------------------------Apply median filter-----------------------------------------#

medBlur = cv2.medianBlur(img, 5)
plt.subplot(121),plt.imshow(medBlur),plt.title('Median filtered')
plt.xticks([]), plt.yticks([])
plt.show()


circles = cv2.HoughCircles(medBlur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, 
param2=10, minRadius=85, maxRadius=86)
# Draw detected circles
if circles is not None:
 circles = np.uint16(np.around(circles))
 for i in circles[0, :]:
     # Draw outer circle
     cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
     # Draw inner circle
     cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

plt.subplot(121),plt.imshow(img),plt.title('Circle Detecting')
plt.xticks([]), plt.yticks([])
plt.show()


#ret,th1 = cv2.threshold(medBlur,127,255,cv2.THRESH_BINARY)

#titles = ['Original Image', 'Global Thresholding (v = 127)',
         #   'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#images = [medBlur, th1]

#for i in range(2):
 #   plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
  #  plt.title(titles[i])
   # plt.xticks([]),plt.yticks([])
#plt.show()


# Change color to RGB (from BGR)
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
plt.imshow(image)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = image.reshape((-1,3))
 
# Convert to float type
pixel_vals = np.float32(pixel_vals)

#the below line of code defines the criteria for the algorithm to stop running,
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy)
#becomes 85%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering wit h number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
k = 3
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
segmented_image = segmented_data.reshape((image.shape))
 
plt.subplot(121),plt.imshow(segmented_image),plt.title('Clustered')
plt.xticks([]), plt.yticks([])
plt.show()