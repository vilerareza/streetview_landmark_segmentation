# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:07:03 2020

@author: Reza Vilera
"""

import numpy as np
import os
from keras import models
from keras import layers
from keras import Input
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
from skimage.io import imshow
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint

nclass = 1
BATCH_SIZE = 20

def psp (input_size = (256,256,3)):
    inputs = Input(input_size, name = 'input')
    
    conv1 = layers.Conv2D(64, 7, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv1') (inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2,2), name = 'pool1') (conv1)
    
    conv2 = layers.Conv2D(64, 7, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv2') (pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2), name = 'pool2') (conv2)
    
    conv3 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv3') (pool2)
    pool3 = layers.MaxPooling2D(pool_size=(2,2), name = 'pool3') (conv3)
        
    conv4 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv4') (pool3)

    # 32x32x64
    
    skip = conv4
    
    conv5 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'conv5') (conv4)
    conv6 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', dilation_rate = (2,2), name = 'conv7') (conv5)
   
    
    add = layers.Add(name = 'add')([conv6, skip])
    
    res = layers.Conv2D(128, 1, activation = 'relu', padding = 'same', 
                          kernel_initializer = 'he_normal', name = 'residual-conv') (add)
    #32x32x128
   
    #pyramid pooling
    pyramid1 = layers.AveragePooling2D(pool_size = (2,2), name = 'pyramid1')(res)
    pyramid2 = layers.AveragePooling2D(pool_size = (4,4), name = 'pyramid2')(res)
    pyramid3 = layers.AveragePooling2D(pool_size = (8,8), name = 'pyramid3')(res)
    pyramid4 = layers.AveragePooling2D(pool_size = (32,32), name = 'pyramid4')(res)
    
    #1x1 convolution
    conv_pyramid1 = layers.Conv2D(32, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid1')(pyramid1)
    up1 = layers.UpSampling2D(size = (2,2), name = 'up1')(conv_pyramid1)
    
    conv_pyramid2 = layers.Conv2D(32, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid2')(pyramid2)
    up2 = layers.UpSampling2D(size = (4,4), name = 'up2')(conv_pyramid2)
    
    conv_pyramid3 = layers.Conv2D(32, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid3')(pyramid3)
    up3 = layers.UpSampling2D(size = (8,8), name = 'up3')(conv_pyramid3)
    
    conv_pyramid4 = layers.Conv2D(32, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid4')(pyramid4)
    up4 = layers.UpSampling2D(size = (32,32), name = 'up4')(conv_pyramid4)
        
    concat = layers.concatenate([res,up1,up2,up3,up4], axis = 3, name = 'concat')

    up_final = layers.Conv2DTranspose(32, kernel_size = (8,8), strides = (8,8),name = 'up_final')(concat)

    conv_output = layers.Conv2D(nclass,1, activation = 'sigmoid', name = 'conv_output', padding='same')(up_final)
    
    model = models.Model(inputs, conv_output)
    
    model.compile(optimizer = optimizers.adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        
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

x = image_datagen.flow(x_train[:int(x_train.shape[0]*0.9)],
                        batch_size = BATCH_SIZE, seed = 42)
y = mask_datagen.flow(y_train[:int(y_train.shape[0]*0.9)],
                        batch_size = BATCH_SIZE, seed = 42)

#validation generator
image_datagen_val = ImageDataGenerator()
mask_datagen_val = ImageDataGenerator()

x_val = image_datagen_val.flow(x_train[int(x_train.shape[0]*0.9):],
                                batch_size = BATCH_SIZE, seed = 42)
y_val = mask_datagen_val.flow(y_train[int(x_train.shape[0]*0.9):],
                                batch_size = BATCH_SIZE, seed = 42)


train_generator = zip(x,y)
val_generator = zip(x_val, y_val)

#earlystopper = EarlyStopping(patience=3, verbose = 1)
checkpointer = ModelCheckpoint('psp-run-save-landmark-final.h5', verbose = 1)

model = psp()

#model = models.load_model('psp-final-landmark-final.h5')

#results = model.fit_generator(train_generator, validation_data = val_generator, validation_steps = 5, steps_per_epoch = 150, epochs = 3, 
#                                  callbacks = [checkpointer])#earlystopper,checkpointer])

# model.save('psp-final-landmark-final.h5')

# # plotting the results
# result_dict = results.history
# acc = result_dict['accuracy']
# val_acc = result_dict['val_accuracy']
# loss = result_dict['loss']
# val_loss = result_dict['val_loss']
# epochs = range(1, len(acc)+1)
# plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
# plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
# plt.xticks([1,2,3])
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend(['Training Accuracy', 'Validation Accuracy'], loc=4)






 
