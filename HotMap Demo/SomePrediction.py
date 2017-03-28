import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import optimizers, callbacks
from scipy import stats
import cv2


label = "0123456789ABCDEFGHIPQRSTUVWXYabcdefghipqrstuvwxyJKLMNOZjklmnoz"
labels = []
for i in label:
    labels.append(i)
labels.append("Final Prediction")
def transfer_label(input):
    output=[]
    for i in range(len(input)):
        if input[i] == -1:
            output.append("NULL")
        else:
            output.append(label[input[i]])
    return output

np.random.seed(1337)  # for reproducibility
batch_size = 96
nb_classes = 62
nb_epoch = 100

# input image dimensions
img_rows, img_cols = 128,128
# number of convolutional filters to use
nb_filters1 = 48
nb_filters2 = 128
nb_filters3 = 192
nb_filters4 = 192
nb_filters5 = 128
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size1 = (11, 11)
kernel_size2 = (5, 5)
kernel_size3 = (3, 3)
kernel_size4 = (3, 3)
kernel_size5 = (3, 3)

input_shape = (img_rows, img_cols,1)

model = Sequential()

model.add(Convolution2D(nb_filters1, kernel_size1[0], kernel_size1[1],
                        border_mode='valid',
                        input_shape=input_shape))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters2, kernel_size2[0], kernel_size2[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))


model.add(Convolution2D(nb_filters3, kernel_size3[0], kernel_size3[1]))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters4, kernel_size4[0], kernel_size4[1]))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters5, kernel_size5[0], kernel_size5[1]))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.load_weights('Fit.h5')

call=callbacks.EarlyStopping(monitor='loss',min_delta=0.001, patience=2, verbose=1, mode='auto')

optimizer=optimizers.Adam(lr=0.00003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
num = 0

image = cv2.imread('Reshape41.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

window = 10
step = 5
i = 0
d = 0
test_prediction = []
image_sequence = 0
prediction_hot = []
header = []
while i < image.shape[1] - window:
    image_sequence +=1
    slide_window = gray[0:image.shape[0] - 1, i:i + window]
    slide_window = cv2.resize(slide_window,(128,128))
    picturename = 'Slide_window{}'.format(image_sequence)
    image_name = 'D:\CAPSTONE\data\predictionImage\{}.jpg'.format(picturename)
    cv2.imwrite(image_name, slide_window)
    d = d + 1
    i = i + step
    slide_window = np.array(slide_window)
    slide_window = slide_window.reshape(1, 128, 128, 1)
    FinalPredict = model.predict(slide_window)
    FinalPredict = np.array(FinalPredict)
    print(FinalPredict, 'prediction{}'.format(image_sequence))
    print(np.argmax(FinalPredict))
    test_prediction.append(np.argmax(FinalPredict))
    FinalPredict = FinalPredict.flatten()
    FinalPredict = FinalPredict.tolist()
    prediction_hot.append(FinalPredict)
    header.append(picturename)

print(test_prediction)
output = transfer_label(test_prediction)
prediction_hot = np.array(prediction_hot).transpose().tolist()
output = np.array(output).transpose().tolist()
print(prediction_hot,'one hot')
prediction_hot.append(output)
df = pd.DataFrame(prediction_hot, index=labels, columns=header)
df.to_csv('one_hot.csv', index_label="Window_Label",index=True)
print(output)