# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:07:58 2015

@author: Hans-Christiaan
"""

from skimage.io import imread
from mahotas.features import surf
from sklearn.cluster import KMeans

import os 
import pylab as pl
import numpy as np
import cv2

TRAIN_PATH = '../train'
IMG_TEST_PATH = '../train/appendicularian_straight/52410.jpg'

def siftFeaturesTest():
    image = imread(IMG_TEST_PATH)
    sift = cv2.SIFT()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    print "Nr. of sift points: ", len(keypoints)        
    
    img = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    pl.imshow(img)
    pl.show()

def compSift(image):
    sift = cv2.SIFT()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return descriptors

# computes all the SIFT-features of the trainingsset
def computeSiftDataset(train_path):
    class_dirs = os.listdir(train_path)[:6]

    all_sift_features = []       

    for class_dir in class_dirs:
        print "Getting features from class ", class_dir
        # get the paths to all of the images in this class
        class_path = os.path.join(train_path, class_dir)
        img_names = os.listdir(class_path)
        img_paths = [os.path.join(class_path,img_name) for img_name in img_names]
        # compute the SIFT-features of all the images in this class
        # and add them all to the bag of features
        all_sift_features.extend(\
            [compSift(imread(img_path)) for img_path in img_paths])
    
    print "\n","Total amount of features =",len(all_sift_features)

    return all_sift_features
    #k = 50
    #k_m = KMeans(init = 'k-means++', n_clusters = k, n_init = 10)
    #k_m.fit(all_sift_features);
    
    
    
    