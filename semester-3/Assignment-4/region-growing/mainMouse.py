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


def main():
    # Load the image.
    img = cv2.imread('KakashiCyberpunk.jpg', cv2.IMREAD_GRAYSCALE)

    # Create a window to display the image and set the mouse callback function.
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse_click, param=img)

    # Display the image.
    cv2.imshow('image', img)

    # Wait for a key press.
    cv2.waitKey(0)

    # Destroy all windows.
    cv2.destroyAllWindows()


def on_mouse_click(event, x, y, flags, param):
    # Check if the left mouse button was clicked.
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the threshold value from the user.
        threshold = cv2.getTrackbarPos('Threshold', 'image')

        # Apply the region-growing algorithm to the image.
        region_mask = region_growing(param, (x, y), threshold)

        # Display the region mask.
        cv2.imshow('region mask', region_mask)

        # Writing the new Image #
        cv2.imwrite("Segmented Region.jpg", region_mask)


def region_growing(img, seed_point, threshold):
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


if __name__ == '__main__':
    main()
