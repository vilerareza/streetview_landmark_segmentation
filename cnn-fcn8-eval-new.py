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
    
    conv6 = layers.Conv2D(1024, 7, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv6')(conv_base.get_layer(name='block5_pool').get_output_at(0))
    conv7 = layers.Conv2D(1024, 1, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv7')(conv6)
      
    #up7 = layers.Conv2DTranspose(512, kernel_size=(2,2), strides = (2,2))(conv7)
    #up7 = layers.Conv2DTranspose(256, kernel_size=(4,4), strides = (4,4))(conv7)
    up7 = layers.UpSampling2D(size=(4,4), interpolation = 'bilinear', name='up_7')(conv7)
    conv7_4 = layers.Conv2D(256, 1, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv7_4')(up7)
    #concat1 = layers.concatenate([up7,conv_base.get_layer(name='block4_pool').get_output_at(0)], axis = 3, name = 'concat1')
    
    #up4 = layers.Conv2DTranspose(256, kernel_size=(2,2), strides = (2,2))(concat1)
    #up4 = layers.Conv2DTranspose(256, kernel_size=(2,2), strides = (2,2))(conv_base.get_layer(name='block4_pool').get_output_at(0))
    up4 = layers.UpSampling2D(size=(2,2), interpolation = 'bilinear', name='up_4')(conv_base.get_layer(name='block4_pool').get_output_at(0))
    conv4_2 = layers.Conv2D(256, 1, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv4_2')(up4)
    #concat2 = layers.concatenate([up4,conv_base.get_layer(name='block3_pool').get_output_at(0)], axis = 3, name = 'concat2')
    
    pool3 = conv_base.get_layer(name='block3_pool').get_output_at(0)
    conv3_1 = layers.Conv2D(256, 1, activation ='relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv3_1')(pool3)
    
    add = layers.Add (name = 'add')([conv7_4, conv4_2, conv3_1])
    
    up_final = layers.UpSampling2D(size=(8,8), interpolation = 'bilinear', name='up_final')(add)
    
    #out = layers.Conv2D(1, 1, activation ='relu', padding = 'same', 
    #                    kernel_initializer = 'he_normal', name = 'out') (up_final)
    
    #out = layers.Activation('sigmoid')(out)
    
    out = layers.Conv2D(nclass,1, activation = 'sigmoid', name = 'conv_output', padding='same')(up_final)
    
    model = models.Model(conv_base.input, out)
    
    model.compile(optimizer = optimizers.SGD(), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.compile(optimizer = optimizers.adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


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

results = model.fit_generator(train_generator, validation_data = val_generator, validation_steps = 5, steps_per_epoch = 100, epochs = 5, 
                                  callbacks = [earlystopper,checkpointer])#earlystopper,checkpointer])

model.save('fcn-eval-new.h5')





