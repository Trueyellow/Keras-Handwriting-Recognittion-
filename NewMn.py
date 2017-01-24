from __future__ import print_function
import numpy as np

import scipy.io as sio
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers,callbacks

np.random.seed(1337)  # for reproducibility
batch_size = 220
nb_classes = 62
nb_epoch = 100

Test_Data=sio.loadmat('Testdata.mat')
Test_Data=np.array(Test_Data.get('Testdata'))

Test_Label=sio.loadmat('TestLabel.mat')
Test_Label=np.array(Test_Label.get('Testlabel'))

Train_Data=sio.loadmat('Traindata.mat')
Train_Data=np.array(Train_Data.get('Traindata'))

Train_Label=sio.loadmat('TrainLabel.mat')
Train_Label=np.array(Train_Label.get('TrainLabel'))

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = (img_rows, img_cols, 1)

Train_Data = Train_Data.reshape(Train_Data.shape[0], img_rows, img_cols, 1)
Test_Data = Test_Data.reshape(Test_Data.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

Train_Data= Train_Data.astype('float32')
Test_Data = Test_Data.astype('float32')

# convert class vectors to binary class matrices
Train_Label = np_utils.to_categorical(Train_Label, nb_classes)
Test_Label = np_utils.to_categorical(Test_Label , nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))

model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))



call=callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')

optimizer=optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(Train_Data, Train_Label, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, callbacks=[call],validation_data=(Test_Data, Test_Label))

score = model.evaluate(Test_Data, Test_Label, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save_weights('Fit.h5')