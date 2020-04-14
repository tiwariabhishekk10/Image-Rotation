import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.utils import to_categorical
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
#from keras.preprocessing.image import ImageDataGenerator

## Reading rotated image

# train

data_train_x=[]
path_train=glob.glob("D:/Python Code/Image_rotation/train/*.jpg")

#reading in orginal form
for img in path_train:
    n=cv2.imread(img)
    n=cv2.resize(n,(64,64))
    data_train_x.append(n)

#Scale the raw pixel 
images_train_x=np.array(data_train_x,dtype='float')/255.0

type(images_train_x)
images_train_x.shape

train_y=pd.read_csv(r"D:\Python Code\Image_rotation\train.csv")
train_y=train_y[train_y.columns[1]]
trainy=np.array(train_y)
trainy=trainy.reshape(-1,1)#reshape to (953,1)

# test

data_test_x=[]
path_test=glob.glob("D:/Python Code/Image_rotation/test/*.jpg")

#reading in orginal form
for img in path_test:
    img = cv2.imread(img)
    n2=cv2.resize(img,(64,64))
    data_test_x.append(n2)

images_test_x=np.array(data_test_x,dtype='float')/255.0
print(type(images_test_x))
images_test_x.shape

test_y=pd.read_csv(r"D:\Python Code\Image_rotation\test.csv")
test_y=test_y[test_y.columns[1]]
testy=np.array(test_y)
testy=testy.reshape(-1,1)

print('train shape x',images_train_x.shape)
print('test shape x',images_test_x.shape)

print('train y',trainy.shape)
print('test y',testy.shape)

trainy=to_categorical(trainy)
trainy
testy=to_categorical(testy)

#image augmentation
#datagen = ImageDataGenerator(height_shift_range=0.5,width_shift_range=0.5,rescale=True,shear_range=0.2,fill_mode='nearest')
#datagen.fit(images_train_x)
#datagen.fit(images_test_x)

model=Sequential()

model.add(Conv2D(16,(3,3),padding='same',activation='relu',input_shape=images_train_x.shape[1:]))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(359,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=90)

#Based on the choice of performance measure, the “mode” argument will need to be specified as whether the objective 
# of the chosen metric is to increase (maximize or ‘max‘) or to decrease (minimize or ‘min‘).
#We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. 
# This can be done by setting the “patience” argument.

mc = ModelCheckpoint('classification.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

#model_fit=model.fit_generator(datagen.flow(images_train_x,train_y,batch_size=32),epochs=100,validation_data=datagen.flow(images_test_x,test_y),shuffle=True)
model_fit=model.fit(images_train_x,trainy,batch_size=64,epochs=100,validation_data=(images_test_x,testy),shuffle=True,callbacks=[mc])
