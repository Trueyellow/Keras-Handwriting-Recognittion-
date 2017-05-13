"""
Rutgers capstone--Team 37
Char_prediction.py
This is a character level prediction program based on VGG-16 net loaded with our trained weights.
We first preprocess our input image by cut it input character level image that we can use those cutted pictures as input
of our prediction network input, and gives us result.
"""

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
import pandas as pd
from preprocess import line_extraction, extract_single_character

DATA_SIZE = (32, 32)

label = "abcdefghipqrstuvwxyjklmnoz"
labels = []


def transfer_label(input):
    output = []
    for i in range(len(input)):
        if input[i] == 27:
            output.append("NULL")
        else:
            output.append(label[input[i]-1])
    return output

def VGGprediction(image):

    np.random.seed(1337)  # for reproducibility

    for i in label:
        labels.append(i)

    labels.append("NULL")

    labels.append("Final Prediction")

    nb_classes = 28

    # input image dimensions
    img_rows, img_cols = DATA_SIZE[0], DATA_SIZE[1]

    input_shape = (img_rows, img_cols, 3)

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='block1_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='block2_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.load_weights('acc0.9804300589370728.h5')

    optimizer = optimizers.Adam(lr=0.00003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Test parameters
    test_prediction = []
    image_sequence = 0
    prediction_hot = []
    header = []

    line = line_extraction(image)
    char, line = extract_single_character(line)
    for slide in char:
        slide = slide.reshape(1, DATA_SIZE[0], DATA_SIZE[1], 3)
        FinalPredict = model.predict(slide)
        FinalPredict = np.array(FinalPredict)
        print(FinalPredict, 'prediction{}'.format(image_sequence))
        print(np.argmax(FinalPredict))
        test_prediction.append(np.argmax(FinalPredict))
        FinalPredict = FinalPredict.flatten()
        FinalPredict = FinalPredict.tolist()
        prediction_hot.append(FinalPredict)
        header.append(image_sequence)

    print(test_prediction)

    string = []
    char_num = 0
    line_num = 0
    print(line)
    for i in test_prediction:
        if i != 27:
            string.append(label[i-1])
            char_num += 1
            if char_num == line[line_num]:
                string.append("LINE")
                char_num = 0
                if line_num < (len(line)-1):
                    line_num += 1
        else:
            string.append("NULL")

    return string
