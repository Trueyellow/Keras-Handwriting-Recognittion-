from __future__ import print_function
import cv2
import numpy as np
from PIL import Image

DATA_SIZE = (32, 32)


def cut(image, isgray=0):
    if isgray == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    hist_x = np.zeros((gray.shape[0], 1))
    hist_y = np.zeros((gray.shape[1], 1))

    for i in range(binary.shape[0]):
        counter = 0
        for j in range(binary.shape[1]):
            if binary[i][j] == 0:
                counter += 1
        hist_x[i] = counter

    for m in range(binary.shape[1]):
        counter = 0
        for n in range(binary.shape[0]):
            if binary[n][m] == 0:
                counter += 1
        hist_y[m] = counter

    left = right = up = down = 0

    for a in range(binary.shape[0]):
        if hist_x[a] != 0:
            up = a
            break

    for b in reversed(range(binary.shape[0])):
        if hist_x[b] != 0:
            down = b
            break

    for c in range(binary.shape[1]):
        if hist_y[c] != 0:
            left = c
            break

    for d in reversed(range(binary.shape[1])):
        if hist_y[d] != 0:
            right = d
            break

    if left - 3 >= 0 and right + 3 < binary.shape[1] and up - 3 >= 0 and down + 3 < binary.shape[0]:
        new_image = image[up - 3:down + 3, left - 3:right + 3]
    else:
        new_image = image[up:down, left:right]

    return new_image


def image_preprocess(image_name, DATA_SIZE):
    image = cv2.imread(image_name)
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    _, thresh = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY)  # threshold

    _, binary_erosion = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY_INV)

    _, contours, hierarchy = cv2.findContours(binary_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    contour_list = []
    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
        # get rectangle bounding contour
        [y, x, w, h] = cv2.boundingRect(contour)
        c = [y, x, w, h]

        # discard areas that are too large
        if h > 300 and w > 300:
            continue

        # discard areas that are too small

        # if h < 5 or w < 5:
        #     continue

        if h * w < 1100:
            continue
        contour_list.append(c)

    contour_list = np.array(contour_list)

    processed_image = []

    for i in range(len(contour_list)):
        min_index = np.argmin(contour_list, axis=0)[0]
        contour_i = contour_list[min_index]
        [y, x, w, h] = contour_i

        contour_list = np.delete(contour_list, min_index, 0)

        if x - 80 > 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
            slide_window = gray1[x - 80:x + h + 2, y - 3:y + w + 3]
        elif x - 80 < 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
            slide_window = gray1[0:x + h, y:y + w]
        else:
            slide_window = gray1[x:x + h, y:y + w]

        new_image = cut(slide_window, 1)

        slide_window = new_image

        slide_window = data_preprocess(slide_window, DATA_SIZE, 1)

        slide_window = slide_window.reshape(1, DATA_SIZE[0], DATA_SIZE[1], 3)

        if processed_image == []:
            processed_image = slide_window
        else:
            processed_image = np.concatenate((processed_image, slide_window), axis=0)

    return processed_image
    # -----------------------------------------------


def data_preprocess(image, size, isgray=0):
    image = cv2.resize(image, size)
    if isgray == 0:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    else:
        gray = image
    gray = gray.reshape(32, 32, 1)
    gray = np.concatenate((gray, gray, gray), axis=2)
    return gray


def line_extraction(picture_name):
    image = cv2.imread(picture_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    medianBlur = cv2.medianBlur(thresh, 3)
    # gauss = cv2.GaussianBlur(thresh, (5, 5), 0)
    d = 0
    y = np.zeros((gray.shape[0], 1))
    for i in range(gray.shape[0]):
        counter = 0
        for j in range(gray.shape[1]):
            if medianBlur[i][j] != 0:
                counter += 1
        y[i] = counter

        line = []
        above_index = []
        below_index = []
        window = 10

        counter = 0
    for index in range(y.shape[0]):

        if counter % 2 == 0:
            if index <= window:
                temp = y[0:index + 10]

            elif index > window and index < y.shape[0] - window - 1:
                temp = y[index - 10:index + 10]

            elif index > y.shape[0] - window + 1:
                temp = y[index:y.shape[0]]

            nozero = np.count_nonzero(temp)
            if y[index] != 0 and nozero > 5:
                above_index.append(index)
                counter += 1

        if counter % 2 == 1:
            if index <= window:
                temp = y[0:index + 10]

            elif index > window and index < y.shape[0] - window - 1:
                temp = y[index - 10:index + 10]

            elif index > y.shape[0] - window + 1:
                temp = y[index:y.shape[0]]

            nozero = np.count_nonzero(temp)
            if y[index] == 0 and nozero > window - 5:
                below_index.append(index)
                counter += 1

    show_image = image

    for i in range(min(len(above_index), len(below_index))):
        if below_index[i] - above_index[i] > 20:
            newImage = image[above_index[i]:below_index[i], 0:image.shape[1]]
            cv2.line(show_image, (0, above_index[i]), (image.shape[1], above_index[i]), (0, 0, 255), 3)
            cv2.line(show_image, (0, below_index[i]), (image.shape[1], below_index[i]), (0, 0, 255), 3)
            # filename = "%d.jpg" % d
            cv2.imwrite("9090_{}.jpg".format(d), newImage)
            d = d + 1
            line.append(newImage)
        else:
            continue

    cv2.imwrite("show_image.jpg",show_image)
    line = np.array(line)

    return line


def extract_single_character(line_list):
    d = 1
    processed_image = []
    char_num = []
    line_num = 0
    d = 0
    for image in line_list:
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        _, thresh = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY)  # threshold
        d += 1
        _, binary_erosion = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('12345{}.jpg'.format(d), binary_erosion)
        _, contours, hierarchy = cv2.findContours(binary_erosion, cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)  # get contours
        contour_list = []
        w_sum = 0
        h_sum = 0

        # For each contour found, append them all into a numpy array
        for contour in contours:
            # get rectangle bounding contour
            [y, x, w, h] = cv2.boundingRect(contour)
            c = [y, x, w, h]
            print(c)
            # # discard areas that are too large
            if h > 300 and w > 300:
                continue

            # discard areas that are too small

            if h < 40:
                continue
            contour_list.append(c)

        if len(contour_list) == 0:
            continue

        contour_list = np.array(contour_list)
        w_ave = w_sum / len(contour_list)
        h_ave = h_sum / len(contour_list)

        for i in range(len(contour_list)):
            if contour_list[i][3] > 3 * h_ave or contour_list[i][2] > 3 * w_ave:
                np.delete(contour_list, i)
            if contour_list[i][3] * contour_list[i][2] < 0.85 * w_ave * h_ave:
                np.delete(contour_list, i)
        sort_y_contour = []

        # Sort y so that the picture is in sequence
        for i in range(len(contour_list)):
            min_index = np.argmin(contour_list, axis=0)[0]
            contour_i = contour_list[min_index]
            sort_y_contour.append(contour_i)
            # [y, x, w, h] = contour_i
            contour_list = np.delete(contour_list, min_index, 0)

        sort_y_contour = np.array(sort_y_contour)
        space_distance = 0
        for i in range(len(sort_y_contour) - 1):
            space_distance += abs(sort_y_contour[i][0] + w - sort_y_contour[i + 1][0])
        ave_space = space_distance / len(sort_y_contour) - 1

        num = 0
        for i in range(len(sort_y_contour)):
            [y, x, w, h] = sort_y_contour[i]
            if x - 80 > 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
                test = gray1[x - 80:x + h + 2, y - 3:y + w + 3]
            elif x - 80 <= 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
                test = gray1[0:x + h, y:y + w]
            else:
                test = gray1[0:gray1.shape[0], y:y + w]

            test = cut(test, 1)
            d = d + 1
            cv2.imwrite('D:\\CAPSTONE\\data\\finaldemo{}.jpg'.format(d), test)
            slide_window = data_preprocess(test, DATA_SIZE, 1)
            num += 1
            slide_window = slide_window.reshape(1, DATA_SIZE[0], DATA_SIZE[1], 3)

            if processed_image == []:
                processed_image = slide_window
            else:
                processed_image = np.concatenate((processed_image, slide_window), axis=0)

            if i != len(sort_y_contour) - 1:
                if sort_y_contour[i + 1][0] - y - w > 3 * ave_space:
                    d = d + 1
                    test = gray1[0:gray1.shape[0], sort_y_contour[i][0] + w:sort_y_contour[i + 1][0]]
                    cv2.imwrite('D:\\CAPSTONE\\data\\finaldemo{}.jpg'.format(d), test)
                    # test = cut(test, 1)
                    slide_window = data_preprocess(test, DATA_SIZE, 1)
                    slide_window = slide_window.reshape(1, DATA_SIZE[0], DATA_SIZE[1], 3)
                    if processed_image == []:

                        processed_image = slide_window
                    else:
                        processed_image = np.concatenate((processed_image, slide_window), axis=0)
        char_num.append(num)
    print(char_num, 'LINE CHAR NUM')
    return processed_image, char_num

def write_text(picturename,result_list):
    show_image=cv2.imread(picturename)
    window=100
    for i in range(len(result_list)):
        cv2.putText(show_image,result_list[i],(0,show_image.shape[1]-window),4,1,(255,0,0),1)
        window=window-20
    cv2.imwrite("show_image1.jpg",show_image)
    show_image=Image.open("show_image.jpg")
    show_image.show()