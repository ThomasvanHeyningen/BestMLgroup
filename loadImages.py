# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:03:04 2015

Work In Progress. Nog niet gebruiken!

@author: Hans-Christiaan
"""

import os
import random
from skimage.io import imread


def loadImages(train_path, frac, class_indices = range(1,121)):
    """ Loads (a fraction of the) images out of (a chunk of) the trainingset.
    Gives back the images and an array of their respective classes.    
    """
    # load (a chunk of) the class directory names              
    class_dirs = os.listdir(train_path)
    class_dirs = [class_dirs[i] for i in class_indices]
    print "Loading paths of classes: ", class_indices 

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
      
        img_paths.extend([os.path.join(class_path,img_name) for img_name in img_names])
        classes.extend([class_n for i in range(0,len(img_names))])
        class_n = class_n + 1
    print "Amount of images =", len(img_paths)    
    
    # read and return the images and return their class names as well   
    return [imread(image_path) for image_path in img_paths], classes

def loadTrainAndTestSet(dataset_path, sample_fraction, test_fraction, class_indices = range(0,121)):
    """ Loads a training- and test set from 'dataset_path'.\n
     -- 'sample_fraction': controls how many weighted samples are taken from the entire dataset (e.g. 0.5 = half of every class). \n
     -- 'class_indices': the indices of the classes for which the images should be loaded from. (Optional argument, standard is all classes (0 to 120).
    """
    
    images, classes = loadImages(dataset_path, sample_fraction, class_indices)    
    
    # randomly take out TEST_FRAC * len(images) samples for testing
    N_img = len(images)    
    test_indices = random.sample(range(0,N_img), int(test_fraction*N_img))    
    
    test_images = [images[i] for i in test_indices]
    test_classes = [classes[i] for i in test_indices]
    
    train_images = [images[i] for i in range(0,N_img) if i not in test_indices]
    train_classes = [classes[i] for i in range(0,N_img) if i not in test_indices]
    
    return train_images, train_classes, test_images, test_classes
    
