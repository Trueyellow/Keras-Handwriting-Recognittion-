"""
Rutgers capstone--Team 37
transfer.py
This is a image preprocessing and transfer model that turns our raw image into NPY data to save and for further prediction
training.
Train data and test data are divied into several part that can help use save GPU memory in VGG net training step.
"""
from __future__ import print_function
import os
from random import shuffle
import cv2
import numpy as np
# from preprocess import cut, data_preprocess

np.random.seed(1337)  # for reproducibility


def cut(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

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


def reshape(data_image):
    data = np.array(data_image)
    data = data.reshape(data.shape[0], 3, 32, 32)
    data = data.astype('float32')
    data /= 255
    return data


Names = ['train', 'validation']

data_image = []
data_label = []

DataList = []

for dirname in os.listdir(Names[0]):  # [1:] Excludes .DS_Store from Mac OS
    path = os.path.join(Names[0], dirname)
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            DataList.append(os.path.join(Names[0], dirname, filename))

shuffle(DataList)

num = 0
size = (32, 32)
counter = 1

for filename in DataList:
    label = filename.split('\\')[1]
    data_label.append(label)
    image = cv2.imread(filename)
    if label == '36':
        image = cv2.resize(image, size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        gray = gray.reshape(32, 32, 1)
        gray = np.concatenate((gray, gray, gray), axis=2)
        data_image.append(gray)
    else:
        image = cut(image)
        image = cv2.resize(image, size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        gray = gray.reshape(32, 32, 1)
        gray = np.concatenate((gray, gray, gray), axis=2)
        data_image.append(gray)

    num += 1
    print(num)
    if num % 100000 == 0:
        data = reshape(data_image)
        np.save('Traindata{}'.format(counter), data)
        labels = np.array(data_label)
        np.save('Trainlabel{}'.format(counter), labels)
        data_image = []
        data_label = []
        counter += 1

data = reshape(data_image)
np.save('Traindata{}'.format(counter),data)
labels = np.array(data_label)
np.save('Trainlabel{}'.format(counter),labels)

Test_image = []
Test_label = []

TestList = []

for dirname in os.listdir(Names[1]):
    path = os.path.join(Names[1], dirname)
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            TestList.append(os.path.join(Names[1], dirname, filename))

shuffle(TestList)

num = 0
counter = 1
for filename in TestList:
    label = filename.split('\\')[1]
    Test_label.append(label)
    image = cv2.imread(filename)
    if label == '36':
        image = cv2.resize(image, size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        gray = gray.reshape(32, 32, 1)
        gray = np.concatenate((gray, gray, gray), axis=2)
        Test_image.append(gray)
    else:
        image = cut(image)
        image = cv2.resize(image,size)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
        gray = gray.reshape(32, 32, 1)
        gray = np.concatenate((gray, gray, gray), axis=2)
        Test_image.append(gray)
    num += 1
    print(num)

    if num % 100000 == 0:
        data = reshape(Test_image)
        np.save('Testdata{}'.format(counter), data)
        labels = np.array(Test_label)
        np.save('Testlabel{}'.format(counter), labels)
        Test_image = []
        Test_label = []
        counter += 1

Test = reshape(Test_image)
Test_labels = np.array(Test_label)
np.save('Testdata{}'.format(counter), Test)
np.save('Testlabel{}'.format(counter), Test_labels)
