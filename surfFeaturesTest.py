# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 22:22:40 2015

    Om de SURF/SIFT-features uit te testen in classifiers.

@author: Hans-Christiaan
"""

import os
import random
from skimage.io import imread
import siftFeaturesExtractor as se
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as met

TRAIN_PATH = '../train'

def loadImages(train_path, class_range_lower, class_range_upper, frac):
    """ Loads (a fraction of the) images out of (a chunk of) the trainingset.
    Gives back the images and an array of their respective classes.            
    """
    # load (a chunk of) the class directory names              
    class_dirs = os.listdir(train_path)[class_range_lower:class_range_upper]
    print "Loading paths of classes", class_range_lower, "till", class_range_upper

    img_paths = []
    classes = []
    class_n = 0
        
    for class_dir in class_dirs:            
        # get the paths to all of the images in this class
        class_path = os.path.join(train_path, class_dir)
        img_names = os.listdir(class_path)
            
        # take a random sample from the images in this class.
        n_samples = min(int(len(img_names) * frac) + 1, len(img_names))
        img_names = random.sample(img_names, n_samples)             
            
        #print "\t", class_dir, "[", len(img_names), "]"           
      
        img_paths.extend([os.path.join(class_path,img_name) for img_name in img_names])
        classes.extend([class_n for i in range(0,len(img_names))])
        class_n = class_n + 1
    print "Amount of images =", len(img_paths)    
    
    return [imread(image_path) for image_path in img_paths], classes

def ClassifierTest():
    """ Tests out classifiers on the SURF-features.    
    """
    train_images, train_classes = loadImages(TRAIN_PATH, 40, 43, 1.0)
    
    sift_extr = se.SiftExtractor()
    sift_extr.clusterFeatures(train_images, 20)
        
    sift_bags = [sift_extr.getBagOfWordsImage(image) for image in train_images]
    
    # Naïve Bayes lijkt slecht te werken naarmate er veel klassen zijn.    
    bayes_classifier = MultinomialNB()
    bayes_classifier.fit(sift_bags, train_classes)
    
    test_imgs, test_classes = loadImages(TRAIN_PATH, 40, 43, 0.2)
    sift_test_bags = [sift_extr.getBagOfWordsImage(image) for image in test_imgs]
    
    
    predictions = bayes_classifier.predict(sift_test_bags)
    
    accuracy = met.accuracy_score(test_classes, predictions)
    
    print "accuracy:", accuracy
    
    return predictions, test_classes

    
    