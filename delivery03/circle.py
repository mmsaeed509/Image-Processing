import sys
import cv2 as cv
import numpy as np


def main():

    src = cv.imread('DetectCirclesExample_10 (1).png', cv.IMREAD_COLOR)
    # Check if image is loaded fine

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    print(cv.HOUGH_GRADIENT)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10,
                              param1=100, param2=30,
                              minRadius=1, maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 2)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 2)

    cv.imshow("detected circles", src)
    cv.waitKey(0)

    return 0


if __name__ == "__main__":
    main()
