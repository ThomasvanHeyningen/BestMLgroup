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
import sklearn.ensemble as ensm
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
    FRAC = 1.0      # fraction of samples used for training and testing
    TEST_FRAC = 0.3 # fraction of the sampled dataset used for testing
    CLASS_LOWER = 0 
    CLASS_UPPER = 5    
    
    images, classes = loadImages(TRAIN_PATH, CLASS_LOWER, CLASS_UPPER, FRAC)    
    
    # randomly take out TEST_FRAC * len(images) samples for testing
    N_img = len(images)    
    test_indices = random.sample(range(0,N_img), int(TEST_FRAC*N_img))    
    
    test_images = [images[i] for i in test_indices]
    test_classes = [classes[i] for i in test_indices]
    
    train_images = [images[i] for i in range(0,N_img) if i not in test_indices]
    train_classes = [classes[i] for i in range(0,N_img) if i not in test_indices]
    
    # extract all the SURF-features and cluster them (vector quantization)        
    sift_extr = se.SiftExtractor()
    sift_extr.clusterFeatures(train_images, 20)
    
    # get the SURF-feature vectors of the images in the training set
    train_feats = [sift_extr.getBagOfWordsImage(image) for image in train_images]
    
    # train classifier on the feature vectors
    random_forest = ensm.RandomForestClassifier(n_estimators = 30)
    random_forest.fit(train_feats, train_classes)
    
    # get the SURF-feature vectors of the images in the test set
    test_feats = [sift_extr.getBagOfWordsImage(image) for image in test_images]
    
    predictions = random_forest.predict_proba(test_feats)
    
    # Logloss score      
    logloss_score = met.log_loss(test_classes, predictions)
 
    print "log loss:", logloss_score 
    
    