#####################################
#                                   #
#  @author      : 00xWolf           #
#    GitHub    : @mmsaeed509       #
#    Developer : Mahmoud Mohamed   #
#  﫥  Copyright : Mahmoud Mohamed   #
#                                   #
#####################################


import numpy as np
import cv2


def region_growing(img, seed_point, threshold):
    # Applies the region-growing algorithm to segment an image.
    #
    # Args:
    #     img (ndarray): The input grayscale image.
    #     seed_point (tuple): The seed point to start the region-growing algorithm from.
    #     threshold (int): The threshold value used to determine whether to add a pixel to the region.
    #
    # Returns:
    #     ndarray: A binary mask indicating the segmented region.

    # Initialize the region mask and the visited flag array.
    region_mask = np.zeros_like(img, dtype=np.uint8)
    visited = np.zeros_like(img, dtype=np.uint8)

    # Initialize the queue with the seed point.
    queue = [seed_point]

    # Get the shape of the image.
    height, width = img.shape

    # Loop until the queue is empty.
    while len(queue) > 0:
        # Pop the next point from the queue.
        current_point = queue.pop(0)

        # Check if the point has already been visited.
        if visited[current_point[1], current_point[0]] == 1:
            continue

        # Mark the point as visited.
        visited[current_point[1], current_point[0]] = 1

        # Check if the point should be added to the region.
        if img[current_point[1], current_point[0]] <= threshold:
            region_mask[current_point[1], current_point[0]] = 255

            # Add the adjacent pixels to the queue.
            neighbors = get_neighbors(current_point, height, width)
            queue.extend(neighbors)

    return region_mask


def get_neighbors(point, height, width):
    #
    # Returns the adjacent pixels to a given point.
    #
    # Args:
    #     point (tuple): The point to get the neighbors of.
    #     height (int): The height of the image.
    #     width (int): The width of the image.
    #
    # Returns:
    #     list: A list of tuples representing the adjacent pixels.

    x, y = point

    neighbors = []

    # Check the top pixel.
    if y > 0:
        neighbors.append((x, y - 1))

    # Check the left pixel.
    if x > 0:
        neighbors.append((x - 1, y))

    # Check the bottom pixel.
    if y < height - 1:
        neighbors.append((x, y + 1))

    # Check the right pixel.
    if x < width - 1:
        neighbors.append((x + 1, y))

    return neighbors


# Load the image.
img = cv2.imread('KakashiCyberpunk.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the region-growing algorithm.
region_mask = region_growing(img, (50, 50), 100)

# Display the results.
cv2.imshow('Input Image', img)
cv2.imshow('Segmented Region', region_mask)
cv2.imwrite("Segmented Region.jpg", region_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
