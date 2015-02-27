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
import time

class SiftExtractor:    
        
    def __init__(self):
        self.TRAIN_PATH = '../train'
        self.IMG_TEST_PATH = '../train/hydromedusae_bell_and_tentacles/60003.jpg'
    
    def showSiftFeatures(self, img_path):
        image = imread(img_path)
        sift = cv2.SURF(1000)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        
        print "Nr. of sift points: ", len(keypoints)        
        
        img = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        pl.imshow(img)
        pl.show()
        return keypoints
    
    def compSift(self,image):
        # OpenCV SURF werkt toch beter: 
        # ongeveer net zoveel lege plaatjes (4,2% t.o.v. 4,8%), 
        # sneller (tot 45% !), 
        # maar wel minder features per plaatje (6,92 t.o.v. 7,85)
        
        # threshold is based on the size of the image:
        # the smaller the image, the smaller the threshold        
        threshold = (len(image) + len(image[0])) / 6
    
        sift = cv2.SURF(threshold)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    
    # Computes all the SIFT-features of the trainingsset and clusters them
    # using K-Means (k clusters).
    # Run this first before running getBagOfWordsImage on an image!
    def computeSiftDataset(self, train_path, k):
        start = time.clock() 
        # for testing purposes, run the algorithm on the first 5 classes              
        class_dirs = os.listdir(train_path)[:10]
        print "Loading paths of all the images..."
        print "classes:"
        img_paths = []
        
        for class_dir in class_dirs:
            print "\t", class_dir
            # get the paths to all of the images in this class
            class_path = os.path.join(train_path, class_dir)
            img_names = os.listdir(class_path)
            img_paths.extend([os.path.join(class_path,img_name) for img_name in img_names]) 
        print "Total amount of images =", len(img_paths)
        print "Extracting and describing SIFT-features of ALL the images!..."
        # get all the descriptors grouped by image (so an array of an array of descriptors)     
        all_sift_features = [self.compSift(imread(img_path)) for img_path in img_paths]        
        before_empty_removal = len(all_sift_features)
        # Remove all empty arrays (images with no SIFT-features :( )        
        all_sift_features = [descr for descr in all_sift_features if descr != None]
        after_empty_removal = len(all_sift_features)
        print "total amount of empty images =", before_empty_removal - after_empty_removal
        # Make one giant list of descriptors (array of N×128)
        all_sift_features = np.concatenate(all_sift_features)
        
        print "\n","Total amount of features found =",len(all_sift_features)
        print "Clustering the features using K-Means...     (with k =", k, ")"
        # clustering using K-Means
        self.k_m = KMeans(init = 'k-means++', n_clusters = k, n_init = 10)
        self.k_m.fit(all_sift_features);
        
        end = time.clock()
        timing = end - start
        print "Centroids computed! Total computing time = ", timing
                       
    
    # Returns a histogram of the amount of descriptors assigned to each cluster.
    def getBagOfWordsImage(self, image):
        sift_features = self.compSift(image)
        predictions = self.k_m.predict(sift_features)
        bins = range(0, len(self.k_m.cluster_centers_))
        return np.histogram(predictions, bins = bins)[0]     