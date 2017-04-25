import cv2
import numpy as np

def line_extraction(picture_name):
    image = cv2.imread(picture_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    medianBlur = cv2.medianBlur(thresh, 3)

    d = 0
    y = np.zeros((gray.shape[0], 1))
    for i in range(gray.shape[0]):
        counter = 0
        for j in range(gray.shape[1]):
            if medianBlur[i][j] != 0:
                counter += 1

    line = []
    above_index=[]
    below_index=[]
    window = 10

    counter=0
    for index in range(y.shape[0]):

        if counter%2==0:
            if index <= window:
                temp=y[0:index+10]

            elif index>window and index<y.shape[0]-window-1:
                temp=y[index-10:index+10]

            elif index>y.shape[0]-window+1:
                temp=y[index:y.shape[0]]

            nozero = np.count_nonzero(temp)
            if y[index] != 0 and nozero > 5:
                above_index.append(index)
                counter+=1

        if counter%2==1:
            if index <= window:
                temp=y[0:index+10]

            elif index>window and index<y.shape[0]-window-1:
                temp=y[index-10:index+10]

            elif index>y.shape[0]-window+1:
                temp=y[index:y.shape[0]]

            nozero = np.count_nonzero(temp)
            if y[index] == 0 and nozero > window - 5:
                below_index.append(index)
                counter+=1

    for i in range(min(len(above_index),len(below_index))):
        newImage = image[above_index[i]:below_index[i], 0:image.shape[1]]
        # filename = "%d.jpg" % d
        cv2.imwrite("9090_{}.jpg".format(d), newImage)
        d = d + 1
        line.append(newImage)

    line = np.array(line)

    return line

picture_name="12121.jpg"

line_extraction(picture_name)