import cv2
import numpy as np

image = cv2.imread("Rutgers.jpeg")
gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
_, thresh = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY)  # threshold
medianBlur = cv2.medianBlur(gray1, 3)
gauss = cv2.GaussianBlur(gray1, (3, 3), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
erosion = cv2.erode(gauss, kernel, iterations=1)

_, binary_erosion = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("Binary_erosion.jpg", binary_erosion)
dilated = cv2.dilate(gauss, kernel, iterations=1)  # dilate"""
close = cv2.erode(dilated, kernel, iterations=4)
open = cv2.dilate(erosion, kernel, iterations=3)

_, contours, hierarchy = cv2.findContours(binary_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

d = 1

contour_list = []

# For each contour found, append them all into a numpy array
for contour in contours:
    # get rectangle bounding contour
    [y, x, w, h] = cv2.boundingRect(contour)
    c = [y, x, w, h]

    # discard areas that are too large
    if h > 300 and w > 300:
        continue

    # discard areas that are too small

    if h < 5 or w < 5:
        continue

    contour_list.append(c)

contour_list = np.array(contour_list)

# Sort y so that the picture is in sequence
for i in range(len(contour_list)):
    min_index = np.argmin(contour_list, axis=0)[0]
    contour_i = contour_list[min_index]
    [y, x, w, h] = contour_i
    contour_list = np.delete(contour_list, min_index, 0)

    if x-5>0 and x+h+2<gray1.shape[0] and y-3>0 and y+w+3<gray1.shape1:
        test = gray1[x - 5:x + h + 2, y - 3:y + w + 3]
    else:
        test=gray1[x:x+h,y:y+w]

    cv2.imwrite("{}.jpg".format(d), test)
    d = d + 1

    # draw rectangle around contour on original image
    # cv2.rectangle(image, (x, y - 30), (x + w, y + h), (255, 0, 0), 2)


cv2.imwrite("Grayscale.jpg", gray1)
cv2.imwrite("GaussBlur.jpeg", gauss)
cv2.imwrite("Binary Image.jpg", thresh)
cv2.imwrite("medianFilter.jpg", medianBlur)
cv2.imwrite("erode_a.jpg", erosion)
cv2.imwrite("dilate_a.jpg", dilated)
cv2.imwrite("open_a.jpg", open)
cv2.imwrite("close_a.jpg", close)
