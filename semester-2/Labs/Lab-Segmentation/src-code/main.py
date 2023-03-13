from skimage import filters
from skimage.data import camera
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = camera()
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

axes[0].imshow(camera(), cmap=plt.cm.gray)
axes[0].set_title('original image')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

# --------------- line detection local processing --------------- #

# define the vertical filter
vertical_filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# define the horizontal filter
horizontal_filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# read in the image
img = plt.imread('img/testLines.jpg')

# get the dimensions of the image
n, m, d = img.shape

# initialize the edges image
edges_img = img.copy()

# loop over all pixels in the image
for row in range(3, n - 2):
    for col in range(3, m - 2):
        # create little local 3x3 box
        local_pixels = img[row - 1:row + 2, col - 1:col + 2, 0]

        # apply the vertical filter
        vertical_transformed_pixels = vertical_filter * local_pixels
        # remap the vertical score
        vertical_score = vertical_transformed_pixels.sum() / 4

        # apply the horizontal filter
        horizontal_transformed_pixels = horizontal_filter * local_pixels
        # remap the horizontal score
        horizontal_score = horizontal_transformed_pixels.sum() / 4

        # combine the horizontal and vertical scores into a total edge score
        edge_score = (vertical_score ** 2 + horizontal_score ** 2) ** .5

        # insert this edge score into the edges image
        edges_img[row, col] = [edge_score] * 3

# write new image #
cv2.imwrite("output-img/local processing line .jpg", edges_img)

cv2.imshow("local processing line", edges_img)
cv2.waitKey(1000)

# --------------- line detection Hough Transform --------------- #

Image_2 = "img/testLines.jpg"
# Loads an image
original_image = cv2.imread(cv2.samples.findFile(Image_2), cv2.IMREAD_GRAYSCALE)

dst = cv2.Canny(original_image, 50, 200, None, 3)
cv2.imshow("detected", dst)

# Copy edges to the images that will display the results in BGR
cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)

# write new image #
cv2.imwrite("output-img/Detected Lines.jpg", cdstP)
cv2.imwrite("output-img/gray.jpg", original_image)

cv2.imshow("original image", original_image)
cv2.waitKey(1000)

cv2.imshow("Detected Lines", cdstP)
cv2.waitKey(1000)

# Apply Edge linking using Hough Transform #

img = cv2.imread('img/sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('output-img/HoughLines.jpg', img)
cv2.imshow("Detected Hough Lines", img)
cv2.waitKey(1000)
