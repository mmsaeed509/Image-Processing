"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main():
    default_file = 'e34ce0452108219af69b5afe90fa0982.jpg'
    filename = "testLines.jpg"
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    dst = cv.Canny(src, 50, 200, None, 3)
    cv.imshow("detcted", dst)
    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines", cdstP)

    cv.waitKey()
    return 0


if __name__ == "__main__":
    main()
