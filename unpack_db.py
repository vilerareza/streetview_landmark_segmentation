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

N_CLUSTERS=10

# Load database
db = pickle.load(open("db.p","rb"))


# Unpack database

# Init 
record_number_db = np.empty((0))
hist_db = np.empty((0,N_CLUSTERS))
coord_db = np.empty((0,2))
# Unpack
for record in db:
    record_number_db = np.append(record_number_db, np.ravel(record[0]), axis=0)
    hist_db = np.append(hist_db, np.reshape(record[1], (1,len(record[1]))), axis=0)
    coord_db = np.append(coord_db, np.reshape(record[2],(1,len(record[2]))), axis=0)

# Fit database for NN
db_nn = KNeighborsClassifier(n_neighbors=1)
db_nn.fit(hist_db,record_number_db)

# Dump
#db_nn = pickle.dump(db_nn, open("db_nn","wb"))
 
