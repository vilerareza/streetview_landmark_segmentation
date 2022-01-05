# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:15:18 2021

@author: Reza Vilera
"""

from sklearn.cluster import KMeans
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from sklearn.neighbors import  KNeighborsClassifier
import pickle


db_image_folder = 'E:/testimages/surabaya-street/database/db_image/'
db_mask_folder = 'E:/testimages/surabaya-street/database/db_mask/'
fileNames = os.listdir(db_image_folder)

bovw_collection = np.empty(0)
hist_db_collection = np.empty(0)
accuracies = np.empty(0)

featureLength = 128
n_iter = 1


for it in range(n_iter):
        
    #Cluster Model  
    
    clusters=10
    
    
    ## Extract features from all images in database    
    features = np.empty([0,128],np.float32)#Collection of SIFT features from all imagees in database
    
    for fileName in fileNames:
        image = cv.imread(db_image_folder+fileName,0)
        mask = cv.imread(db_mask_folder+fileName,0) 
        image = cv.normalize(image,None,0,255,cv.NORM_MINMAX)
        
        #FAST detector
        fast1 = cv.FastFeatureDetector_create(threshold=10)
        kp1 = fast1.detect(image,mask)
        #SIFT extractor
        sift1 = cv.xfeatures2d_SIFT.create()
        kp1, des1 = sift1.compute(image, kp1)

        features = np.append(features, des1, axis=0)
        
    
    # Bag of Visual Words
    
    ## Clustering    
    cluster_center = pickle.load(open("cluster_center90.p","rb"))        
    bovw = KMeans (n_clusters = clusters, n_init=10, init = cluster_center, max_iter = 1000)
    bovw.fit(features)
    
    bovw_collection = np.append(bovw_collection, bovw)
    
    # Create histogram for database
    hist_db  = np.empty([0,clusters]) # init histogram collection from all images in database
    
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
    
    ## Fit database for NN    
    knn_label = range(1,len(hist_db)+1) #Label for NN
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(hist_db,knn_label)
 
    
    # Test image    
    test_image_folder = 'E:/testimages/surabaya-street/test/test_image/'
    test_mask_folder = 'E:/testimages/surabaya-street/test/test_mask/'
    # test_image_folder = 'E:/testimages/surabaya-street/database/db_image/'
    # test_mask_folder = 'E:/testimages/surabaya-street/database/db_mask/'
    
    # test file name
    test_files = ['31.jpg','21.jpg','28.jpg','32.jpg','34.jpg','41.jpg','44.jpg','46.jpg','52.jpg','55_1.jpg']
    # Expected match record on database
    truth = np.array([9,26,7,6,13,8,10,11,25,17]) 
    
    result  = np.empty(0)
    
    i=0
    for test_file in test_files:
        test_image = cv.imread(test_image_folder+test_file,0)
        test_image = cv.normalize(test_image,None,0,255,cv.NORM_MINMAX)
        test_mask = cv.imread(test_mask_folder+test_file,0) 
        
        #FAST detector
        fast_test = cv.FastFeatureDetector_create(threshold=10)
        kp_test = fast_test.detect(test_image,test_mask)
        #SIFT extractor
        sift_test = cv.xfeatures2d_SIFT.create()
        kp_test, des_test = sift_test.compute(test_image, kp_test)
        #Feature cluster assignment
        clust_test = bovw.predict(des_test)
        #Feature histogram creation
        hist_test, edge = np.histogram(clust_test, bins = clusters)
        
        #match with database
        match = knn.predict([hist_test])
        #collect match result
        result = np.append(result, match)
        
    correct = result==truth
    accuracy = sum(correct)/10
    
    accuracies = np.append(accuracies, accuracy)


acc_max = np.max(accuracies)
acc_max_index = np.argmax(accuracies)

best_bovw = bovw_collection[acc_max_index]

pickle.dump(best_bovw,open("bovw.p", "wb"))

