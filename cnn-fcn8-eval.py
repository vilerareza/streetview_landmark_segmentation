# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:39:56 2020

@author: Reza Vilera
"""
import numpy as np
import os
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from skimage.io import imshow
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.applications import VGG16
#from keras.applications import ResNet101

BATCH_SIZE = 5
nclass = 1

def fcn8():
        
    n= 4096
    
    conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (256,256,3))
    
    conv6 = layers.Conv2D(n, 8, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv6')(conv_base.get_layer(name='block5_pool').get_output_at(0))
    conv7 = layers.Conv2D(n, 1, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv7')(conv6)
    
    conv7_4 = layers.Conv2DTranspose(nclass,kernel_size = (4,4), strides = (4,4), name = 'conv7_4')(conv7)
    conv411 = layers.Conv2D(nclass, 1, activation  = 'relu', padding = 'same', name = 'conv411')(conv_base.get_layer(name='block4_pool').get_output_at(0))
    conv411_2 = layers.Conv2DTranspose(nclass, kernel_size = (2,2), strides = (2,2), name = 'conv422_2')(conv411)
    conv311 = layers.Conv2D(nclass, 1, activation = 'relu', padding = 'same', name = 'conv311')(conv_base.get_layer(name = 'block3_pool').get_output_at(0))
    
    out = layers.Add (name = 'add')([conv411_2, conv311, conv7_4])
    out = layers.Conv2DTranspose(nclass, kernel_size=(8,8), strides = (8,8))(out)
    out = layers.Activation('sigmoid')(out)
    
    model = models.Model(conv_base.input, out)
    
    #model.compile(optimizer = optimizers.SGD(), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.compile(optimizer = optimizers.adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return conv_base


image_folder = 'E:/testimages/cityscapes-image-pairs/frames/'
mask_landmark = 'E:/testimages/cityscapes-image-pairs/landmark/'


def get_img (path):
    names = os.listdir(path)
    data = []
    for name in names:
        #img = cv.imread((path + name), cv.IMREAD_GRAYSCALE)
        img = cv.imread(path + name)
        img = img [:,:,::-1]
        img = cv.normalize(img, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
        #img = img.reshape(256,256,1)
        data.append(img)
        del img
    del names
    return data 

def get_msk (path1):
    names = os.listdir(path1)
    data = []
    
    for name in names:
        img = cv.imread((path1 + name),cv.IMREAD_GRAYSCALE)
        img = cv.normalize(img, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
        img = img.reshape(256,256,1)
        data.append(img)
        del img
    del names
    return data 

x_train = np.array(get_img(image_folder))
y_train = np.array(get_msk(mask_landmark))


# training generator
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()

x = image_datagen.flow(x_train[:500],
                        batch_size = BATCH_SIZE, seed = 42)
y = mask_datagen.flow(y_train[:500],
                        batch_size = BATCH_SIZE, seed = 42)

#validation generator
image_datagen_val = ImageDataGenerator()
mask_datagen_val = ImageDataGenerator()

x_val = image_datagen_val.flow(x_train[500:600],
                                batch_size = BATCH_SIZE, seed = 42)
y_val = mask_datagen_val.flow(y_train[500:600],
                                batch_size = BATCH_SIZE, seed = 42)

train_generator = zip(x,y)
val_generator = zip(x_val, y_val)

earlystopper = EarlyStopping(patience=3, verbose = 1)
checkpointer = ModelCheckpoint('fcn-run-save.h5', verbose = 1)

model = fcn8()

#model = models.load_model('psp-final-landmark-final.h5')

#results = model.fit_generator(train_generator, validation_data = val_generator, validation_steps = 5, steps_per_epoch = 100, epochs = 5, 
#                                  callbacks = [earlystopper,checkpointer])#earlystopper,checkpointer])

#model.save('fcn-eval.h5')





