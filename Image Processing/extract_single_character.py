import cv2
import numpy as np

# -------------- cut function --------------------
def cut(gray):
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    binary = np.array(binary)
    print(binary.shape)
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
        new_image = gray[up - 3:down + 3, left - 3:right + 3]
    else:
        new_image = gray[up:down, left:right]

    return new_image


def extract_single_character(line_list):

    d=1
    character=[]
    for image in line_list:
        # image = cv2.imread("line0.jpg")
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        _, thresh = cv2.threshold(gray1, 90, 255, cv2.THRESH_BINARY)  # threshold
        medianBlur = cv2.medianBlur(gray1, 3)
        gauss = cv2.GaussianBlur(gray1, (3, 3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        erosion = cv2.erode(gauss, kernel, iterations=1)

        _, binary_erosion = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite("Binary_erosion_cd.jpg", binary_erosion)
        dilated = cv2.dilate(thresh, kernel, iterations=1)  # dilate"""
        close = cv2.erode(dilated, kernel, iterations=4)
        open = cv2.dilate(erosion, kernel, iterations=3)


        _, contours, hierarchy = cv2.findContours(binary_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours


        contour_list = []
        w_sum = 0
        h_sum = 0
        # For each contour found, append them all into a numpy array
        for contour in contours:
            # get rectangle bounding contour
            [y, x, w, h] = cv2.boundingRect(contour)
            c = [y, x, w, h]
            w_sum += w
            h_sum += h
            # # discard areas that are too large
            # if h > 300 and w > 300:
            #     continue
            #
            # # discard areas that are too small
            #
            # if h * w <1100:
            #     continue

            contour_list.append(c)

        contour_list = np.array(contour_list)
        w_ave = w_sum / len(contour_list)
        h_ave = h_sum / len(contour_list)
        print(w_ave, 'w_ave')
        print(h_ave, 'h_ave')

        for i in range(len(contour_list)):
            if contour_list[i][3] > 3 * h_ave or contour_list[i][2] > 3 * w_ave:
                np.delete(contour_list, i)
            if contour_list[i][3] * contour_list[i][2] < 0.85 * w_ave * h_ave:
                np.delete(contour_list, i)

        print(contour_list, 'contor')
        sort_y_contour = []

        # Sort y so that the picture is in sequence
        for i in range(len(contour_list)):
            min_index = np.argmin(contour_list, axis=0)[0]
            contour_i = contour_list[min_index]
            sort_y_contour.append(contour_i)
            # [y, x, w, h] = contour_i
            contour_list = np.delete(contour_list, min_index, 0)

        sort_y_contour = np.array(sort_y_contour)
        print(sort_y_contour)
        space_distance = 0
        for i in range(len(sort_y_contour) - 1):
            space_distance += abs(sort_y_contour[i][0] + w - sort_y_contour[i + 1][0])
        ave_space = space_distance / len(sort_y_contour) - 1
        print(ave_space, 'ave')

        for i in range(len(sort_y_contour)):
            [y, x, w, h] = sort_y_contour[i]
            if x - 80 > 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
                test = gray1[x - 80:x + h + 2, y - 3:y + w + 3]
            elif x - 80 <= 0 and x + h + 2 < gray1.shape[0] and y - 3 > 0 and y + w + 3 < gray1.shape[1]:
                test = gray1[0:x + h, y:y + w]
            else:
                test = gray1[x:x + h, y:y + w]

            test = cut(test)

            cv2.imwrite("space{}.jpg".format(d), test)
            d = d + 1
            if i != len(sort_y_contour) - 1:
                if sort_y_contour[i + 1][0] - y - w > 1.5 * ave_space:
                    test = gray1[0:gray1.shape[0], sort_y_contour[i][0] + w:sort_y_contour[i + 1][0]]
                    cv2.imwrite("space{}.jpg".format(d), test)
                    d=d+1

        character.append(test)
    character=np.array(character)

    return character

                    # draw rectangle around contour on original image
                    # cv2.rectangle(image, (x, y - 30), (x + w, y + h), (255, 0, 0), 2)

# cv2.imwrite("Grayscale_cd.jpg", gray1)
# cv2.imwrite("GaussBlur_cd.jpeg", gauss)
# cv2.imwrite("Binary Image_cd.jpg", thresh)
# cv2.imwrite("medianFilter_Cd.jpg", medianBlur)
# cv2.imwrite("erode_a_cd.jpg", erosion)
# cv2.imwrite("dilate_a_cd.jpg", dilated)
# cv2.imwrite("open_a_cd.jpg", open)
# cv2.imwrite("close_a_cd.jpg", close)