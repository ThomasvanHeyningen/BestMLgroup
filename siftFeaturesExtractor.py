# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:07:58 2015

@author: Hans-Christiaan
"""

from skimage.io import imread
from sklearn.cluster import KMeans

import os 
import pylab as pl
import numpy as np
import cv2

class SiftExtractor:    
        
    def __init__(self):
        self.TRAIN_PATH = '../train'
        self.IMG_TEST_PATH = '../train/appendicularian_straight/52410.jpg'
    
    def showSiftFeatures(self, img_path):
        image = imread(img_path)
        sift = cv2.SIFT()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        print "Nr. of sift points: ", len(keypoints)        
        
        img = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pl.imshow(img)
        pl.show()
    
    def compSift(self,image):
        sift = cv2.SIFT()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    
    # Computes all the SIFT-features of the trainingsset and clusters them
    # using K-Means (k clusters).
    # Run this first before running getBagOfWordsImage on an image!
    def computeSiftDataset(self, train_path, k):
        # for testing purposes, run the algorithm on the first 5 classes
        class_dirs = os.listdir(train_path)[:5]
    
        img_paths = []
    
        for class_dir in class_dirs:
            print "Getting features from class ", class_dir
            # get the paths to all of the images in this class
            class_path = os.path.join(train_path, class_dir)
            img_names = os.listdir(class_path)
            img_paths.extend([os.path.join(class_path,img_name) for img_name in img_names]) 
        # get all the descriptors grouped by image (so an array of an array of descriptors)
        all_sift_features = [self.compSift(imread(img_path)) for img_path in img_paths]
        # Remove all empty arrays (images with no SIFT-features :( )
        all_sift_features = [descr for descr in all_sift_features if descr != None]
        # Make one giant list of descriptors (array of N×128)
        all_sift_features = np.concatenate(all_sift_features)
        
        print "\n","Total amount of features =",len(all_sift_features)
        print "Clustering the features using K-Means, with k =", k
        # clustering using K-Means
        self.k_m = KMeans(init = 'k-means++', n_clusters = k, n_init = 10)
        self.k_m.fit(all_sift_features);               
    
    # Returns a histogram of the amount of descriptors assigned to each cluster.
    def getBagOfWordsImage(self, image):
        sift_features = self.compSift(image)
        predictions = self.k_m.predict(sift_features)
        bins = range(0, len(self.k_m.cluster_centers_))
        return np.histogram(predictions, bins = bins)[0]     