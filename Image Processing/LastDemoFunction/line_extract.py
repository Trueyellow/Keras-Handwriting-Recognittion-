import cv2
import numpy as np


def line_extraction(picture_name):
    image = cv2.imread(picture_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    cv2.imwrite('gray2.jpg', gray)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('binary2.jpg', thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=3)
    cv2.imwrite('dilated2.jpg', dilated)

    d = 0
    y = np.zeros((gray.shape[0], 1))
    for x in range(gray.shape[0]):
        y[x] = sum(dilated[x])

    def getWindow(index):
        min_above = y[index]
        min_below = y[index]
        min_above_index = index
        min_below_index = index
        if index > 100 and index < image.shape[0]:
            for i in range(index - 100, index):
                if y[i] < min_above:
                    min_above = y[i]
                    min_above_index = i
            for i in range(index, index + 101):
                if y[i] < min_below:
                    min_below = y[i]
                    min_below_index = i

        elif index < 100:
            for i in range(0, index):
                if y[i] < min_above:
                    min_above = y[i]
                    min_above_index = i
            for i in range(index, index + 101):
                if y[i] < min_below:
                    min_below = y[i]
                    min_below_index = i

        elif index < 100:
            for i in range(index - 101, index):
                if y[i] < min_above:
                    min_above = y[i]
                    min_above_index = i
            for i in range(index, image.shape[0]):
                if y[i] < min_below:
                    min_below = y[i]
                    min_below_index = i

        window = [min_above_index, min_below_index]
        return window

    line = []
    window = 50
    for index in range(y.shape[0]):

        if index < window:
            y1 = y[0:index + window]
        else:
            y1 = y[index - window:index + window]

        maxValue = max(y1)

        if y[index] > 2000 and y[index] == maxValue:
            Window = getWindow(index)
            print(Window)

            newImage = image[Window[0]:Window[1], 0:image.shape[0]]
            # filename = "%d.jpg" % d
            cv2.imwrite("{}.jpg".format(d), newImage)
            d = d + 1
            line.append(newImage)

    print(image.shape)
    cv2.imwrite('lines2.jpg', image)
    line = np.array(line)

    return line

picture_name="multiline.jpg"

line_extraction(picture_name)