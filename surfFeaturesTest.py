# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 22:22:40 2015

    Om de SURF/SIFT-features uit te testen in classifiers.

@author: Hans-Christiaan
"""


import random
import time
import numpy as np

import skimage.transform as tr

import featureExtraction as fe
import siftFeaturesExtractor as se
import loadImages as limg
import featureExtractionNew as fen

import warnings

import sklearn.multiclass as mlc
import sklearn.svm as svm
import sklearn.ensemble as ensm
import sklearn.metrics as met

TRAIN_PATH = '../train'
# preprocessing lijkt geen invloed op de performance te hebben
#TRAIN_PATH = '../train_preprocessed'

def resizeImages(images, size):
    return [tr.resize(img,(size, size)) for img in images]

# Dit vreet al het geheugen op, zelfs als er maar op twee klassen getrained wordt...
def extendWithOtherFeatures(images, sift_features, img_max_size):
    images_resized = resizeImages(images, img_max_size)
    
    other_feats_extractor = fe.featureExtractor(img_max_size, len(images_resized))
    other_feats = other_feats_extractor.extract(images_resized, None)
    
    return [np.append(a, b) for a in sift_features for b in other_feats]

def computeFeatureVector(surf_extr, image):
    # get the vector quantified SURF-features
    surf_quant = surf_extr.getBagOfWordsImage(image)
    # get the region properties
    reg_props = fen.getRegionFeatures(image)    
    
    
    return np.append(surf_quant, reg_props)

def ClassifierTest():
    """ Tests out classifiers on the SURF-features.    
    """
    start = time.clock()
    
    FRAC = 0.4      # fraction of the dataset used for training and testing
    TEST_FRAC = 0.2 # fraction of the sampled dataset used for testing
    CLUSTER_FRAC = 0.05 # fraction of the sampled training set used to cluster the SURF-features
    CLASS_LOWER = 0 
    CLASS_UPPER = 121    
    
    train_images, train_classes, test_images, test_classes = \
                        limg.loadImages(TRAIN_PATH, CLASS_LOWER, CLASS_UPPER, FRAC, TEST_FRAC)    
        
    # randomly take out CLUSTER_FRAC * len(train_images) samples for clustering
    # the SURF-features.
    N_train_imgs = len(train_images)
    cluster_indices = random.sample(range(0,N_train_imgs), int(CLUSTER_FRAC*N_train_imgs))    
    
    cluster_images = [train_images[i] for i in cluster_indices]
    
    print "Clustering on the SIFT-features of", len(cluster_images), "images."
    # extract all the SURF-features and cluster them (vector quantization) 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       
        sift_extr = se.SiftExtractor()
        sift_extr.clusterFeatures(cluster_images, 20)
    
    # get the SURF-feature vectors of the images in the training set
    #train_feats = [sift_extr.getBagOfWordsImage(image) for image in train_images]
    
    train_feats = [computeFeatureVector(sift_extr, image) for image in train_images]    
    
    #img_size = 40   
    # extend the SURF-features with the other features.                       
    #train_feats = extendWithOtherFeatures(train_images, train_feats, img_size)
    
    # train classifier on the feature vectors
    
    #classifier = mlc.OutputCodeClassifier(ensm.RandomForestClassifier(n_estimators = 30),\
        #code_size = 2, random_state = 0)   
         
    #classifier = mlc.OneVsOneClassifier(ensm.RandomForestClassifier(n_estimators = 30))
    
    # This classifier works the best for now, with an accuracy of around  0.75 ~ 0.796, [HC]     
    classifier = mlc.OneVsRestClassifier(ensm.RandomForestClassifier(n_estimators = 30))
       
    print "training the classifier on the vector-quantified SIFT-features..."
    classifier.fit(train_feats, train_classes)
    
    print "predicting the classes of the test set..."
    # get the SURF-feature vectors of the images in the test set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #test_feats = [sift_extr.getBagOfWordsImage(image) for image in test_images]
        test_feats = [computeFeatureVector(sift_extr, image) for image in test_images]        
        
    # extend the SURF-features with the other features.
    #test_feats = extendWithOtherFeatures(test_images, test_feats, img_max_size)
    
    #predictions_proba = classifier.predict_proba(test_feats)
    predictions = classifier.predict(test_feats)    
    
    accuracy = met.accuracy_score(test_classes, predictions)    
    print "accuracy:", accuracy
     
    #logloss_score = met.log_loss(test_classes, predictions_proba) 
    #print "log loss:", logloss_score 
    
    end = time.clock()
    timing = end - start
    print "Total computation took", timing/60, "minutes"   