# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 11:07:58 2015

@author: Hans-Christiaan
"""

from sklearn.cluster import KMeans
 
import pylab as pl
import numpy as np
import cv2
import time

class SiftExtractor:    
        
    def __init__(self):       
        self.IMG_TEST_PATH = '../train/hydromedusae_bell_and_tentacles/60003.jpg'
    
    def showSiftFeatures(self, image):        
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
    def computeSiftDataset(self, images):
        """ Computes all the SIFT-features of [images]
        """
        
        print "Extracting and describing SIFT-features of the images..."
        # get all the descriptors grouped by image (so an array of an array of descriptors)     
        all_sift_features = [self.compSift(image) for image in images]        
        before_empty_removal = len(all_sift_features)
        # Remove all empty arrays (images with no SIFT-features :( )        
        all_sift_features = [descr for descr in all_sift_features if descr != None]
        after_empty_removal = len(all_sift_features)
        print "Amount of empty images =", before_empty_removal - after_empty_removal
        # Make one giant list of descriptors (array of N×128)
        all_sift_features = np.concatenate(all_sift_features)
                
        print "\n","Amount of features found =",len(all_sift_features)
        
        return all_sift_features           
        
        
    def clusterFeatures(self, images, k_clusters):
        """ Computes all the features of [images]
        and clusters them using K-Means.            
            k_clusters -- the amount of clusters
        """
        start = time.clock()        
        
        # get (a sample of) the sift-features of the dataset
        surf_features = self.computeSiftDataset(images)
        # cluster the features using K-Means
        print "Clustering features using K-means (k =",k_clusters,")"
        self.k_m = KMeans(init='k-means++', n_clusters=k_clusters, n_init=10)
        self.k_m.fit(surf_features)
        print "Features clustered!"
        
        end = time.clock()
        timing = end - start
        print "Clustering took", timing/60, "minutes"        

    def getBagOfWordsImage(self, image):
        """ Gets the histogram of an image based on cluster membership.
        Run [clusterFeatures] first to compute the clusters!
            image -- the image for which the histogram needs to be calculated
        """
        bins = range(0, len(self.k_m.cluster_centers_))
        
        sift_features = self.compSift(image)
        if sift_features != None :
            predictions = self.k_m.predict(sift_features)
            return np.histogram(predictions, bins = bins)[0]
        else :
            return np.zeros(len(self.k_m.cluster_centers_)-1).astype(int)