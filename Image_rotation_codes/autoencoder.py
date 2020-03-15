import cv2
import glob
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import matplotlib.pyplot as plt

from keras.models import Model
from keras import regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
from keras.engine.input_layer import Input
from numpy.random.mtrand import randint
 
def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#####Train

data_train_x=[]
path_train_x=glob.glob("D:/Python Code/Image_rotation/Auto Encoder/trainx/*.jpg")

for img in path_train_x:
    n=cv2.imread(img)
    n=cv2.resize(n,(224,224))
    data_train_x.append(n)
 
images_train_x=np.array(data_train_x,dtype='float')/255.0
images_train_x.shape

data_train_y=[]
path_train_y=glob.glob("D:/Python Code/Image_rotation/Auto Encoder/trainy/*.jpg")

for img in path_train_y:
    n1=cv2.imread(img)
    n1=cv2.resize(n1,(224,224))
    data_train_y.append(n1)

images_train_y=np.array(data_train_y,dtype='float')/255.0
images_train_y.shape

#####Test

data_test_x=[]
path_test_x=glob.glob("D:/Python Code/Image_rotation/Auto Encoder/testx/*.jpg")

for img in path_test_x:
    n2=cv2.imread(img)
    n2=cv2.resize(n2,(224,224))
    data_test_x.append(n2)

images_test_x=np.array(data_test_x,dtype='float')/255.0
images_test_x.shape

data_test_y=[]
path_test_y=glob.glob("D:/Python Code/Image_rotation/Auto Encoder/testy/*.jpg")

for img in path_test_y:
    n22=cv2.imread(img)
    n22=cv2.resize(n22,(224,224))
    data_test_y.append(n22)

images_test_y=np.array(data_test_y,dtype='float')/255.0
images_test_y.shape

#####Autoencoder

class Autoencoder():
    def __init__(self):
        self.img_rows=224
        self.img_cols=224
        self.channels=3
        self.img_shape=(self.img_rows,self.img_cols,self.channels)

        self.autoencoder_model=self.build_model()
        self.autoencoder_model.compile(loss='mse',optimizer='adam',metrics=[rmse])
        self.autoencoder_model.summary()

    def build_model(self):
        input_layer=Input(shape=self.img_shape)
        #encoder
        h=Conv2D(32,(3,3),activation='relu',padding='same')(input_layer)
        h=MaxPooling2D(pool_size=(2, 2),padding='same')(h)
        h=Conv2D(16,(3,3),activation='relu',padding='same')(h)
        h=MaxPooling2D(pool_size=(2, 2),padding='same')(h)
        #decoder
        h=Conv2D(16,(3,3),activation='relu',padding='same')(h)
        h=UpSampling2D((2,2))(h)
        h=Conv2D(32,(3,3),activation='relu',padding='same')(h)
        h=UpSampling2D((2,2))(h)
        output_layer=Conv2D(3,(3,3),activation='sigmoid',padding='same')(h)

        return Model(input_layer,output_layer)

    def train_model(self, images_train_x,images_train_y,images_test_x,images_test_y,epochs,batch_size):
        history=self.autoencoder_model.fit(images_train_x,images_train_y, batch_size=batch_size,epochs=epochs,validation_data=(images_test_x,images_test_y))

    def eval_model(self,images_test_x):
        preds=self.autoencoder_model.predict(images_test_x)
        return preds

ae=Autoencoder()
ae.train_model(images_train_x,images_train_y,images_test_x,images_test_y,epochs=1,batch_size=20)

preds=ae.eval_model(images_test_x)
