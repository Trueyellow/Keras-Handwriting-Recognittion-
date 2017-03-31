import cv2
import numpy as np

image = cv2.imread('Capstone.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray2.jpg', gray)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('binary2.jpg', binary)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(binary, kernel, iterations=3)
cv2.imwrite('dilated2.jpg', dilated)

d = 0
# counter = 0
y = np.zeros((gray.shape[0], 1))
for i in range(gray.shape[0]):
    counter=0
    for j in range(gray.shape[1]):
        if binary[i][j] != 0:
            counter += 1
    y[i] = counter


def getWindow(index):
    min_above = y[index]
    min_below = y[index]
    for i in range(index - 30, index):
        if y[i] < min_above:
            min_above = y[i]
    for i in range(index, index + 31):
        if y[i] < min_below:
            min_below = y[i]
    window = [min_above, min_below]
    return window


window = 15
for index in range(y.shape[0]):

    if index < window:
        y1 = y[0:index + window]
    else:
        y1 = y[index - window:index + window]

    maxValue = max(y1)

    if y[index] > 200 and y[index] == maxValue:
        print(index)
        Window = getWindow(index)
        newImage = image[index - Window[0]:index + Window[1], 0:gray.shape[1]]
        cv2.imwrite('line{}.jpg'.format(d), newImage)
        d = d + 1
        # cv2.line(image, (0, index), (image.shape[0], index), (0, 0, 255), 1)


        # cv2.line(image, (0, index), (y[index] / 1000, index), (255, 0, 0), 1)

# print(y)
# print(y.shape)
# cv2.imwrite('lines2.jpg', image)
#
# y1 = y[20:25]
# minValue = min(y1)
# print(y1)
# print(minValue)
