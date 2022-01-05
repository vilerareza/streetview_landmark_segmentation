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
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.applications import ResNet101

nclass = 1
BATCH_SIZE = 5

def psp (input_size = (256,256,3)):

    conv_base = ResNet101(weights = 'imagenet', include_top = False, input_shape = input_size)
    #8x8x2048
    resnet_out = (conv_base.get_layer(name='conv5_block3_out').get_output_at(0))
   
    #pyramid pooling
    pyramid1 = layers.AveragePooling2D(pool_size = (1,1), name = 'pyramid1')(resnet_out)
    pyramid2 = layers.AveragePooling2D(pool_size = (2,2), name = 'pyramid2')(resnet_out)
    pyramid3 = layers.AveragePooling2D(pool_size = (4,4), name = 'pyramid3')(resnet_out)
    pyramid4 = layers.AveragePooling2D(pool_size = (8,8), name = 'pyramid4')(resnet_out)
    
    #1x1 convolution
    conv_pyramid1 = layers.Conv2D(512, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid1')(pyramid1)
    #up1 = layers.UpSampling2D(size = (1,1), name = 'up1')(conv_pyramid1)
    
    conv_pyramid2 = layers.Conv2D(512, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid2')(pyramid2)
    up2 = layers.UpSampling2D(size = (2,2), name = 'up2')(conv_pyramid2)
    
    conv_pyramid3 = layers.Conv2D(512, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid3')(pyramid3)
    up3 = layers.UpSampling2D(size = (4,4), name = 'up3')(conv_pyramid3)
    
    conv_pyramid4 = layers.Conv2D(512, 1,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name = 'conv_pyramid4')(pyramid4)
    up4 = layers.UpSampling2D(size = (8,8), name = 'up4')(conv_pyramid4)
        
    concat = layers.concatenate([resnet_out,conv_pyramid1,up2,up3,up4], axis = 3, name = 'concat')

    up_final = layers.Conv2DTranspose(32, kernel_size = (32,32), strides = (32,32),name = 'up_final')(concat)

    conv_output = layers.Conv2D(nclass,1, activation = 'sigmoid', name = 'conv_output', padding='same')(up_final)
    
    model = models.Model(conv_base.input, conv_output)
    
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
checkpointer = ModelCheckpoint('psp-resnet-run-save-landmark-final.h5', verbose = 1)

model = psp()

# #model = models.load_model('psp-final-landmark-final.h5')

results = model.fit_generator(train_generator, validation_data = val_generator, validation_steps = 5, steps_per_epoch = 100, epochs = 5, 
                                  callbacks = [earlystopper,checkpointer])#earlystopper,checkpointer])

model.save('psp-resnet-short-eval.h5')

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






 
