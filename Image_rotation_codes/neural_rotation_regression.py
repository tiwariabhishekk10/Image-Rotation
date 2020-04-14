import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import math

from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
 
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

## Reading rotated image

# train

data_train_x=[]
path_train=glob.glob("D:/Python Code/Image_rotation/Train/*.jpg")

for img in path_train:
    n=cv2.imread(img)
    n=cv2.resize(n,(32,32))
    data_train_x.append(n)

#Scale the raw pixel 
images_train_x=np.array(data_train_x,dtype='float')/255.0

type(images_train_x)
images_train_x.shape

train_y=pd.read_csv(r"D:\Python Code\Image_rotation\rotated_angle_data\Train.csv")
train_y=train_y[train_y.columns[1]]
trainy=np.array(train_y)
trainy=trainy.reshape(-1,1)#reshape to (953,1)

# test

data_test_x=[]
path_test=glob.glob("D:/Python Code/Image_rotation/Test/*.jpg")

for img in path_test:
    n2=cv2.imread(img)
    n2=cv2.resize(n2,(32,32))
    data_test_x.append(n2)

images_test_x=np.array(data_test_x,dtype='float')/255.0
print(type(images_test_x))
images_test_x.shape

test_y=pd.read_csv(r"D:\Python Code\Image_rotation\rotated_angle_data\Test.csv")
test_y=test_y[test_y.columns[1]]
testy=np.array(test_y)
testy=testy.reshape(-1,1)

print('train shape x',images_train_x.shape)
print('test shape x',images_test_x.shape)

print('train y',trainy.shape)
print('test y',testy.shape)

model=Sequential()

model.add(Conv2D(16,(3,3),padding='same',input_shape=images_train_x.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(512)) 
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(32))        
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(1))
model.add(Activation('linear'))

model.compile(loss='mse',optimizer='adam',metrics=[rmse])
model.summary()

model_fit=model.fit(images_train_x,trainy,batch_size=32,epochs=10,validation_data=(images_test_x,testy),shuffle=True),#callbacks=[es])
