"""
Rutgers capstone--Team 37
LSTM_train.py
This is a word sequence level prediction--LSTM neural network training program, we use some opensource novel to train our
LSTM to get better prediction in sequence level
We first preprocess our novel into 3 sequenced-character-input and 1 word label train data, then use those data to train our LSTM and
get character sequence level prediction.
"""
from __future__ import print_function
import numpy
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import codecs
import random
from LSTM_prediction import prediction

# load ascii text and covert to lowercase
filename = "generator.txt"
raw_text = codecs.open(filename).read().lower()

only_character = []

for i in raw_text:
    if i.isalpha():
        only_character.append(i)

# create mapping of unique chars to integers
chars = sorted(list(set(only_character)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
numpy.save('c_to_i_lstm.npy', char_to_int)
int_to_char = dict((i, c) for i, c in enumerate(chars))
numpy.save('i_to_c_lstm.npy', int_to_char)
# summarize the loaded data
n_chars = len(only_character)
n_vocab = len(chars)
print(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 3
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = only_character[i:i + seq_length]
    seq_out = only_character[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

optimizer = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# define the checkpoint
model.load_weights('weights-improvement-05-2.1738.hdf5')
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
# model.fit(X, y, epochs=100, batch_size=512, callbacks=callbacks_list, verbose=1)

for i in range(50):
    predict_sequence = ''
    start_index = random.randint(0, n_patterns - seq_length - 1)
    character_pre_sequence = only_character[start_index: start_index + seq_length]

    for char in character_pre_sequence:
        predict_sequence += char

    predict_sequence += prediction(predict_sequence)

    print('The generate result is: ', predict_sequence)
