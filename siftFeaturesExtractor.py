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
import random
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
        """ Computes the SURF-features of [image]
        """        
        # threshold is based on the size of the image:
        # the smaller the image, the smaller the threshold        
        threshold = (len(image) * len(image[0])) * 0.01
    
        sift = cv2.SURF(threshold)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    
    # Computes the SIFT-features of a chunk of the trainingsset.
    # 
    # [frac] controls how many images have to be sampled from the classes (1 = all images). 
    def computeSiftDataset(self, train_path, class_range_lower, class_range_upper, frac):
        """ Computes the SIFT-features of (a chunk of) the trainingsset. \n
        The chunk starts at class [class_range_lower] and ends at [class_range_upper].\n
        [frac] is the fraction of the image that need to be sampled from the chunk.
        """
        # load (a chunk of) the class directory names              
        class_dirs = os.listdir(train_path)[class_range_lower:class_range_upper]
        print "Loading paths of classes", class_range_lower, "till", class_range_upper

        img_paths = []
        
        for class_dir in class_dirs:            
            # get the paths to all of the images in this class
            class_path = os.path.join(train_path, class_dir)
            img_names = os.listdir(class_path)
            
            # take a random sample from the images in this class.
            n_samples = int(len(img_names) * frac) + 1
            img_names = random.sample(img_names, n_samples)             
            
            #print "\t", class_dir, "[", len(img_names), "]"            
            
            img_paths.extend([os.path.join(class_path,img_name) for img_name in img_names]) 
        print "Amount of images =", len(img_paths)
        print "Extracting and describing SIFT-features of the images..."
        # get all the descriptors grouped by image (so an array of an array of descriptors)     
        all_sift_features = [self.compSift(imread(img_path)) for img_path in img_paths]        
        before_empty_removal = len(all_sift_features)
        # Remove all empty arrays (images with no SIFT-features :( )        
        all_sift_features = [descr for descr in all_sift_features if descr != None]
        after_empty_removal = len(all_sift_features)
        print "Amount of empty images =", before_empty_removal - after_empty_removal
        # Make one giant list of descriptors (array of N×128)
        all_sift_features = np.concatenate(all_sift_features)
                
        print "\n","Amount of features found =",len(all_sift_features)
        
        return all_sift_features
                       
    def computeAndSaveAllFeatures(self):
        chunk_size = 10
        chunk_ranges = [(n*chunk_size, n*chunk_size+chunk_size) for n in range(0,(121/chunk_size)+1)]
        fraction = 1.0
        
        for chunk_range in chunk_ranges:
            chunk = self.computeSiftDataset(self.TRAIN_PATH,\
                chunk_range[0], chunk_range[1], fraction, 3)
            with open("../SIFT_features.csv",'a') as file_handle :
                np.savetxt(file_handle, chunk, delimiter = ";")        
    
    def clusterFeatures(self, sample_fraction, k_clusters):
        """ Computes (a fraction of) the features of the dataset 
        and clusters them using K-Means.
            sample_fraction -- the fraction of the dataset that needs to be sampled \n
            k_clusters -- the amount of clusters
        """
        start = time.clock()        
        
        # get (a sample of) the sift-features of the dataset
        surf_features = self.computeSiftDataset(self.TRAIN_PATH, 0, 121, sample_fraction)
        # cluster the features using K-Means
        print "Clustering features using K-means (k =",k_clusters,")"
        self.k_m = KMeans(init='k-means++', n_clusters=k_clusters, n_init=10)
        self.k_m.fit(surf_features)
        print "Features clustered!"
        
        end = time.clock()
        timing = end - start
        print "Computing time = ", timing        

    def getBagOfWordsImage(self, image):
        """ Gets the histogram of an image based on cluster membership.
        Run [clusterFeatures] first to compute the clusters!
            image -- the image for which the histogram needs to be calculated
        """
        sift_features = self.compSift(image)
        predictions = self.k_m.predict(sift_features)
        bins = range(0, len(self.k_m.cluster_centers_))
        return np.histogram(predictions, bins = bins)[0]     