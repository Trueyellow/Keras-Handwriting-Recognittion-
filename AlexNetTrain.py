import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers,callbacks


batch_size = 96
nb_classes = 62
nb_epoch = 100

TrainDataName =['Traindata1.npy','Traindata2.npy','Traindata3.npy','Traindata4.npy','Traindata5.npy','Traindata6.npy',
        'Traindata7.npy','Traindata8.npy','Traindata9.npy','Traindata10.npy','Traindata11.npy','Traindata12.npy',
        'Traindata13.npy','Traindata14.npy','Traindata15.npy','Traindata16.npy','Traindata17.npy','Traindata18.npy',
        'Traindata19.npy','Traindata20.npy','Traindata21.npy','Traindata22.npy','Traindata23.npy']

TrainLabelName=['Trainlabel1.npy','Trainlabel2.npy','Trainlabel3.npy','Trainlabel4.npy','Trainlabel5.npy','Trainlabel6.npy',
        'Trainlabel7.npy','Trainlabel8.npy','Trainlabel9.npy','Trainlabel10.npy', 'Trainlabel11.npy','Trainlabel12.npy',
        'Trainlabel13.npy','Trainlabel14.npy','Trainlabel15.npy','Trainlabel16.npy', 'Trainlabel17.npy','Trainlabel18.npy',
        'Trainlabel19.npy','Trainlabel20.npy','Trainlabel21.npy','Trainlabel22.npy', 'Trainlabel23.npy']

TestDataName= ['Testdata1.npy','Testdata2.npy','Testdata3.npy','Testdata4.npy'
              ,'Testdata5.npy','Testdata6.npy','Testdata7.npy']

TestLabelName=['Testlabel1.npy','Testlabel2.npy','Testlabel3.npy',
               'Testlabel4.npy','Testlabel5.npy','Testlabel6.npy','Testlabel7.npy']

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

# convert class vectors to binary class matrices

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

model.load_weights('95.h5')

call=callbacks.EarlyStopping(monitor='loss',min_delta=0.001, patience=2, verbose=1, mode='auto')

optimizer=optimizers.Adam(lr=0.00003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
num = 0

for epoch in range(30):

    for trainname in TrainDataName:
        num += 1
        Train_Data = np.load(trainname)
        trainlabel = TrainLabelName[TrainDataName.index(trainname)]
        Train_Label=np.load(trainlabel)
        Train_Data = Train_Data.reshape(Train_Data.shape[0], img_rows, img_cols, 1)
        Train_Label = np_utils.to_categorical(Train_Label, nb_classes)
        model.fit(Train_Data, Train_Label, batch_size=batch_size, nb_epoch=1,
            verbose=2, callbacks=[call])
        print(trainname, "The name of the file")

    model.save_weights('Fit.h5')
    print('Weight Saved')

    for testname in TestDataName:
        Test_Data=np.load(testname)
        Testlabel=TestLabelName[TestDataName.index(testname)]
        Test_Label=np.load(Testlabel)
        Test_Data = Test_Data.reshape(Test_Data.shape[0], img_rows, img_cols, 1)
        Test_Label = np_utils.to_categorical(Test_Label, nb_classes)
        score = model.evaluate(Test_Data, Test_Label, batch_size=batch_size,verbose=0)
        print(score[1])
    print('Epoch ',epoch)
