from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers

train_datagen = ImageDataGenerator(
        rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 196
nb_classes = 62
nb_epoch = 20

train_generator = train_datagen.flow_from_directory(
        'train',# this is the target directory
        class_mode='categorical',
        target_size=(28, 28),  # all images will be resized to 28x28
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        'validation',
        class_mode='categorical',
        target_size=(28, 28),
        batch_size=batch_size)

print(train_generator,"tr")
print(validation_generator,"te")

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
input_shape = (img_rows, img_cols, 3)

Train = np_utils.to_categorical(train_generator, nb_classes)
Test = np_utils.to_categorical(validation_generator, nb_classes)

print(Train,"tr")
print(Test,"te")
"""
train_generator = train_generator.reshape(train_generator.shape[0], img_rows, img_cols, 1)
validation_generator = validation_generator.reshape(validation_generator.shape[0], img_rows, img_cols, 1)


train_generator= train_generator.astype('float32')
validation_generator = validation_generator.astype('float32')
print('train_generator shape:', train_generator.shape)
print(train_generator.shape[0], 'train samples')
print(validation_generator.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
"""
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

optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(Train, Test, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=validation_generator)

model.save_weights('first_tryBatchsize64.h5')

