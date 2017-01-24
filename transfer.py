from __future__ import print_function
import os
from random import shuffle
import cv2
import numpy as np
import scipy.io as sio
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

Names = ['train','validation']

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
for filename in DataList:

    label = filename.split('\\')[1]

    data_label.append(label)

    image = cv2.imread(filename)

    size = (28, 28)

    image = cv2.resize(image, size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

    data_image.append(gray)
    num += 1
    print(num)

data=np.array(data_image)
print(data, 'DATAIMAGE')
labels=np.array(data_label)
print(labels, 'DATALabel')
sio.savemat('Traindata.mat', {'Traindata':data})
sio.savemat('TrainLabel.mat', {'TrainLabel':labels})


Test_image = []
Test_label = []

TestList = []

for dirname in os.listdir(Names[1]):  # [1:] Excludes .DS_Store from Mac OS
    path = os.path.join(Names[1], dirname)
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            TestList.append(os.path.join(Names[1], dirname, filename))

shuffle(TestList)

num = 0

for filename in TestList:
    label = filename.split('\\')[1]

    Test_label.append(label)

    image = cv2.imread(filename)

    size = (28, 28)

    image = cv2.resize(image, size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

    Test_image.append(gray)
    num += 1
    print(num)

Test = np.array(Test_image)
print(Test, 'TestIMAGE')
Test_labels = np.array(Test_label)
print(Test_labels, 'TestLabel')
sio.savemat('Testdata.mat', {'Testdata':Test})
sio.savemat('TestLabel.mat', {'Testlabel':Test_labels})