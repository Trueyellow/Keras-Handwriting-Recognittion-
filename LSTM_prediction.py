"""
Rutgers capstone--Team 37
LSTM_prediction.py
A LSTM prediction class based on trained weights that help us to do final prediction.
"""
from __future__ import print_function
import numpy
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

def lstmprediction(input):

    char_to_int = numpy.load('c_to_i_lstm.npy').item()
    int_to_char = numpy.load('i_to_c_lstm.npy').item()
    model = Sequential()
    model.add(LSTM(256, input_shape=(3, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))

    optimizer = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    # define the checkpoint
    model.load_weights('weights-improvement-05-2.1738.hdf5')

    print('The LSTM input characters are: "' + input + '"')
    x = [char_to_int[char] for char in input]
    x = numpy.reshape(x, (1, 3, 1))
    x = x/float(30)
    preds = model.predict(x, verbose=0)[0]
    next_index = numpy.argmax(preds)
    next_char = int_to_char[next_index]
    return next_char
