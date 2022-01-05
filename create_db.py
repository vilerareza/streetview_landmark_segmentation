# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:15:18 2021

@author: Reza Vilera
"""

from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import  KNeighborsClassifier
import pickle


db_image_folder = 'E:/testimages/surabaya-street/database/db_image/'
db_mask_folder = 'E:/testimages/surabaya-street/database/db_mask/'
fileNames = os.listdir(db_image_folder)

featureLength = 128
clusters=10

# Create histogram for database
hist_db  = np.empty([0,clusters]) # init histogram collection from all images in database

# Load BOVW
bovw = pickle.load(open("bovw.p","rb"))

for fileName in fileNames:    
    image = cv.imread(db_image_folder+fileName,0)
    image = cv.normalize(image,None,0,255,cv.NORM_MINMAX)
    mask = cv.imread(db_mask_folder+fileName,0) 
    
    # #FAST detector
    fast_db = cv.FastFeatureDetector_create(threshold=10)
    kp_db = fast_db.detect(image,mask)
    #SIFT extractor
    sift_db = cv.xfeatures2d_SIFT.create()
    kp_db, des_db = sift_db.compute(image, kp_db)
    #predict cluster for each descriptor
    clust = bovw.predict(des_db)
    #make histogram for each image
    hist, edge = np.histogram(clust, bins=clusters)
    
    #collect for database histogram
    hist_db = np.append(hist_db, np.reshape(hist,(1,clusters)), axis=0) 
    
# Pack database to dictionary
db = {}

record_number = range(1,len(hist_db)+1) #record number for database
coord = pickle.load(open("coord.p","rb")) #coordinate for database

# Database record format (record_number, histogram, coordinate, image file name)
for r in record_number:
    db[r] = (hist_db[r-1], coord[r-1], fileNames[r-1])
    
#db = list(zip(record_number, hist_db, coord, fileNames))

# Fit database for NN
db_nn = KNeighborsClassifier(n_neighbors=1)
db_nn.fit(hist_db,record_number)

# Dump
#pickle.dump(db_nn, open("db_nn.p","wb"))
#pickle.dump(db,open("db.p", "wb"))

