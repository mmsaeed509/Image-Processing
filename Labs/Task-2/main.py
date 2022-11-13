# # # # # # # # # # # # # # # # # # # # # # # #
#
# 﫥  @author   : 00xWolf
#   GitHub    : @mmsaeed509
#   Developer : Mahmoud Mohamed
#
# # # # # # # # # # # # # # # # # # # # # # # #
import cv2
import numpy

# read the img #
img = cv2.imread('tokyo.png')
img2 = cv2.imread('tokyo.png')

# to Distributed pixels to all grey levels #
xp = [0, 64, 128, 192, 255]  # range values for x-axis  #
yp = [0, 16, 128, 240, 255]  # range values for y-axis  #

x = numpy.arange(256)

table = numpy.interp(x, xp, yp).astype('uint8')
print(table)

# lockup table #
img = cv2.LUT(img, table)
cv2.imshow("original img", img2)
cv2.waitKey(3000)
cv2.imshow("stretching img", img)
cv2.waitKey(3000)

cv2.imwrite("tokyo-stretch.png", img)  # write new img #

# cv2.destroyAllWindows()
