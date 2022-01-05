# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 09:07:43 2020

@author: Reza Vilera
"""

#Predict landmark segmentation

import numpy as np
import os
from keras import models
import cv2 as cv
from skimage.io import imshow
from matplotlib import pyplot as plt
from skimage import morphology as morph

model_name = 'psp-landmark-final-overfit-3.h5'
#model_name = 'psp-landmark-final.h5'
#model_name = 'fcn-eval-new.h5'


#image_folder = 'E:/testimages/surabaya-street/frames/training/'
#image_folder = 'E:/testimages/surabaya-street/frames/validation/'
image_folder = 'E:/testimages/cityscapes-image-pairs/frames/'
#image_folder = 'E:/testimages/surabaya-street/database/db_image/'
#image_folder = 'E:/testimages/surabaya-street/test/test_image/'
file_name = '1.jpg'
#label = 0

test_real = cv.imread(image_folder + file_name)
test_real = test_real [:,:,::-1]
test_real = cv.normalize(test_real, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
test_real = cv.resize(test_real, (256,256))
#test_img = cv.imread((image_folder + file_name),cv.IMREAD_GRAYSCALE)
test_img = cv.imread(image_folder + file_name)
test_img = test_img [:,:,::-1]
test_img = cv.normalize(test_img, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
test_img = cv.resize(test_img, (256,256))
#test_img = cv.blur(test_img,(5,5))
input_img = np.reshape(test_img,(1,256,256,3))
#input_img = test_img.reshape(1,256,256,1)
        
model = models.load_model(model_name)

# Perform landmark segmentation (preds)
preds = model.predict(input_img, verbose=1)

# Output thresholding and smoothing
avg = sum(sum(preds[0,:,:,0]))/ (256*256) #average
thres = avg+avg*0.5     #threshold
preds[0,:,:,0] = preds[0,:,:,0] > thres
preds[0,:,:,0] = morph.area_opening(preds[0,:,:,0],200)
preds[0,:,:,0] = morph.area_closing(preds[0,:,:,0],200)

segmentation_mask = (np.squeeze(preds[0,:,:,0])*255).astype(dtype=np.uint8)

pred_output = np.squeeze(preds[0,:,:,0])*255
mask = np.zeros((256,256,3))
mask = cv.normalize(mask, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
mask[:,:,:] = 1.0

fig,ax = plt.subplots(1,2) 
#ax[0].imshow(test_img,cmap = 'gray', vmin = 0, vmax = 1 )
ax[0].imshow(pred_output,cmap = 'gray', vmin = 0, vmax = 1)

it = np.nditer (pred_output, flags = ['multi_index'])
for x in it:
    if (x==255):
        mask[it.multi_index[0], it.multi_index[1],:] = [0,0.4,1]

blend = cv.addWeighted(test_real,0.5, mask, 0.5,0)
ax[1].imshow(blend)

#cv.imwrite((os.path.join('E:/testimages/surabaya-street/map/','5.jpg')), pred_output)