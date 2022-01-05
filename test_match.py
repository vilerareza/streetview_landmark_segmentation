# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:15:18 2021

@author: Reza Vilera
"""
import os
from keras import models
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology as morph
import pickle


# Load

# Segmentation model
model_name = 'psp-landmark-final-overfit-3.h5'
model = models.load_model(model_name)
# Database
db = pickle.load(open("db.p","rb"))
db_nn = pickle.load(open("db_nn.p", "rb"))    
# Bag of Visual Words
bovw = pickle.load(open("bovw.p","rb"))


# Test image

test_file='41.jpg'

test_image_folder = 'E:/testimages/surabaya-street/test/test_image/'
test_mask_folder = 'E:/testimages/surabaya-street/test/test_mask/'
db_image_folder = 'E:/testimages/surabaya-street/database/db_image/'

# Read and normalize

# Color
test_img = cv.imread(test_image_folder + test_file)
test_img_rgb = test_img [:,:,::-1]
test_img_rgb = cv.normalize(test_img_rgb, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
# Gray
#test_img_gray = cv.cvtColor(test_img_rgb, cv.COLOR_RGB2GRAY)
#test_img_gray = cv.normalize(test_img_gray,None,0,255,cv.NORM_MINMAX, dtype=0)
test_img_gray = cv.imread(test_image_folder+test_file,0)
test_img_gray = cv.normalize(test_img_gray,None,0,255,cv.NORM_MINMAX)

# Segmentation

# Adjust shape for prediction
test_img_rgb = np.reshape(test_img_rgb,(1,256,256,3))
# Perform landmark segmentation (pred)
preds = model.predict(test_img_rgb, verbose=1)
# Output thresholding and smoothing
avg = sum(sum(preds[0,:,:,0]))/ (256*256) #average
thres = avg+avg*0.5     #threshold
preds[0,:,:,0] = preds[0,:,:,0] > thres
preds[0,:,:,0] = morph.area_opening(preds[0,:,:,0],200)
preds[0,:,:,0] = morph.area_closing(preds[0,:,:,0],200)

segmentation_mask = (np.squeeze(preds[0,:,:,0])*255).astype(dtype=np.uint8)


#Selective Feature Extraction

# FAST detector
fast_test = cv.FastFeatureDetector_create(threshold=10)
kp_test = fast_test.detect(test_img_gray,segmentation_mask)
# SIFT extractor
sift_test = cv.xfeatures2d_SIFT.create()
kp_test, des_test = sift_test.compute(test_img_gray, kp_test)
# Feature cluster assignment
clust_test = bovw.predict(des_test)
# Create histogram
hist_test, edge = np.histogram(clust_test, bovw.n_clusters)
# Match
match = db_nn.predict([hist_test])


# Display
# Match image in db
match_db_file = db[int(match)][2]
match_db_image = cv.imread(db_image_folder+match_db_file)
match_db_image = match_db_image [:,:,::-1]


fig, ax = plt.subplots(1,3)
ax[0].imshow(np.reshape(test_img_rgb, (256,256,3)))
keypoints_img = cv.drawKeypoints(test_img_gray,kp_test,1)
ax[1].imshow(keypoints_img, cmap = 'gray', vmin=0, vmax=255)
#ax[2].imshow(segmentation_mask, cmap = 'gray', vmin=0, vmax=255)
ax[2].imshow(match_db_image, vmin=0, vmax=255)