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

import numpy as np
import warnings

import sklearn.multiclass as mlc
import sklearn.svm as svm
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
    CLUSTER_FRAC = 0.2 # fraction of the sampled training set used to cluster the SURF-features
    CLASS_LOWER = 0 
    CLASS_UPPER = 121    
    
    images, classes = loadImages(TRAIN_PATH, CLASS_LOWER, CLASS_UPPER, FRAC)    
    
    # randomly take out TEST_FRAC * len(images) samples for testing
    N_img = len(images)    
    test_indices = random.sample(range(0,N_img), int(TEST_FRAC*N_img))    
    
    test_images = [images[i] for i in test_indices]
    test_classes = [classes[i] for i in test_indices]
    
    train_images = [images[i] for i in range(0,N_img) if i not in test_indices]
    train_classes = [classes[i] for i in range(0,N_img) if i not in test_indices]
    
    # randomly take out CLUSTER_FRAC * len(train_images) samples for clustering
    # the SURF-features.
    N_train_imgs = len(train_images)
    cluster_indices = random.sample(range(0,N_train_imgs), int(CLUSTER_FRAC*N_train_imgs))    
    
    cluster_images = [train_images[i] for i in cluster_indices]
    
    # extract all the SURF-features and cluster them (vector quantization) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       
        sift_extr = se.SiftExtractor()
        sift_extr.clusterFeatures(cluster_images, 20)
    
    # get the SURF-feature vectors of the images in the training set
    train_feats = [sift_extr.getBagOfWordsImage(image) for image in train_images]
    
    # train classifier on the feature vectors
    
    #classifier = mlc.OutputCodeClassifier(ensm.RandomForestClassifier(n_estimators = 30),\
        #code_size = 2, random_state = 0)   
         
    #classifier = mlc.OneVsOneClassifier(ensm.RandomForestClassifier(n_estimators = 30))
    
    # This classifier one works the best for now, with an accuracy of 0.288 [HC]     
    classifier = mlc.OneVsRestClassifier(ensm.RandomForestClassifier(n_estimators = 30))
       
    train_feats = np.array(train_feats)
    train_classes = np.array(train_classes)    
    
    print "training the classifier on the vector-quantified SIFT-features..."
    classifier.fit(train_feats, train_classes)
    
    print "predicting the classes of the test set..."
    # get the SURF-feature vectors of the images in the test set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_feats = [sift_extr.getBagOfWordsImage(image) for image in test_images]
    
    #predictions_proba = classifier.predict_proba(test_feats)
    predictions = classifier.predict(test_feats)    
    
    accuracy = met.accuracy_score(test_classes, predictions)    
    print "accuracy:", accuracy
     
    #logloss_score = met.log_loss(test_classes, predictions_proba) 
    #print "log loss:", logloss_score 
    
    