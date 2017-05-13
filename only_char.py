"""
Rutgers capstone--Team 37
Only_char.py
This is a VGG16 net which only trained with words data that imporved our perfomance of English word character-level
prediction.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
import keras
from keras import optimizers, callbacks

np.random.seed(1337)  # for reproducibility
batch_size = 192
nb_classes = 28
nb_epoch = 100

#--------------------------------- call back class--------------------------
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.acc = logs.get('acc')

#--------------------------------- call back class--------------------------



TrainDataName =['Traindata1.npy','Traindata2.npy','Traindata3.npy', 'Testdata1.npy']

TrainLabelName=['Trainlabel1.npy','Trainlabel2.npy','Trainlabel3.npy', 'Testlabel1.npy']

TestDataName= ['Testdata2.npy']

TestLabelName=['Testlabel2.npy']

img_rows, img_cols = 32, 32

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

# model.load_weights("vgg16.h5", by_name=True)
model.load_weights("acc0.9774700572586059.h5")

call = LossHistory()

optimizer = optimizers.Adam(lr=0.00003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


num = 0

lastepoch_acc = 0

# Test_Label = np.load('Testlabel1.npy')

# def preidict():
#     Test_Data = np.load('Testdata1.npy')
#     Test_Label = np.load('Testlabel1.npy')
#     print(Test_Label)
#     Test_Data = Test_Data.reshape(Test_Data.shape[0], img_rows, img_cols, 3)
#     Test_hat = model.predict_classes(Test_Data)
#     Test_Label2 = Test_Label.ravel()
#     Confision = pd.crosstab(Test_hat, Test_Label2)
#     Confision.to_csv('test_data_hotmap.csv')
#
# preidict()
call = LossHistory()
for epoch in range(nb_epoch):
    for trainname in TrainDataName:
        num += 1
        Train_Data=np.load(trainname)
        trainlabel=TrainLabelName[TrainDataName.index(trainname)]
        Train_Label=np.load(trainlabel)
        Train_Data = Train_Data.reshape(Train_Data.shape[0], img_rows, img_cols, 3)
        Train_Label = np_utils.to_categorical(Train_Label, nb_classes)
        model.fit(Train_Data, Train_Label, batch_size=batch_size, epochs=1,
            verbose=2, callbacks=[call])
        print(trainname, "The name of the file")

    lastepoch_acc = call.acc

    model.save_weights('acc{}.h5'.format(lastepoch_acc))

    for testname in TestDataName:
        Test_Data=np.load(testname)
        Testlabel=TestLabelName[TestDataName.index(testname)]
        Test_Label=np.load(Testlabel)
        Test_Data = Test_Data.reshape(Test_Data.shape[0], img_rows, img_cols, 3)
        Test_Label = np_utils.to_categorical(Test_Label, nb_classes)
        score = model.evaluate(Test_Data, Test_Label, batch_size=batch_size, verbose=0)
        print(score[0], score[1])
    print('Epoch ', epoch)

