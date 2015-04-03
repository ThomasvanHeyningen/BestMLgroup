# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 16:47:12 2015

@author: Hans-Christiaan
"""

import os
import numpy as np
from skimage.io import imread
import skimage.transform as tr

TRAIN_PATH = '../train'

def augmentDataset(images, required_size):    
    # sample from image,s with replacement, until the required size is reached
    new_images = np.random.choice(images, required_size)
    # randomly rotate all the images (only in straight angles)
    for img in new_images:
        rot_angle = np.random.choice([0,45,90,135])       
        img = tr.rotate(img,rot_angle)
        
    return new_images    

def underSample(image_paths, required_size):    
    
    new_image_paths = np.random.choice(image_paths, required_size)
    
    return [imread(img_path) for img_path in new_image_paths]

def loadImages(train_path, class_range_lower = 0, class_range_upper = 121, frac = 1, test_frac = 0.2):
    # load (a chunk of) the class directory names              
    class_dirs = os.listdir(train_path)[class_range_lower:class_range_upper]
    print "Loading paths of classes", class_range_lower, "till", class_range_upper

    img_paths_per_class = {}
    img_paths = []
    images = []
    classes = []
    
    # load the image names per class in a dictionary with the class names as keys.    
    for class_dir in class_dirs:            
        # get the paths to all of the images in this class
        class_path = os.path.join(train_path, class_dir)
        img_names = os.listdir(class_path)   

        img_paths_per_class[class_dir] = [os.path.join(class_path,img_name)\
                                            for img_name in img_names]
    
    max_class_size = max([len(imgs) for imgs in img_paths_per_class.itervalues()])  
    
    class_size = int(max_class_size*frac)
    N_classes = class_range_upper-class_range_lower
    print "Uniform class size =", class_size, "images"
    print "Total length dataset =", class_size, "*", N_classes, "=", class_size * N_classes, "images"     
    
    # class size is fixed, so repeat each class name class size times to get
    # the class names for all images
    classes = np.concatenate([np.repeat(class_name, class_size)\
                for class_name in img_paths_per_class.iterkeys()])
        
    for img_paths in img_paths_per_class.itervalues():                
        if len(img_paths) < class_size:
            # actual size is smaller than required size: augment this class
            class_images = [imread(img_path) for img_path in img_paths]
            class_images = augmentDataset(class_images, class_size)            
        if len(img_paths) > class_size:
            # actual size is bigger than required size: sample from this class
            class_images = underSample(img_paths, class_size)
        if len(img_paths) == class_size:
            # actual size is the required size: load all the images
            class_images = [imread(img_path) for img_path in img_paths]
        images = np.append(images, class_images)

    # randomly take out test_frac * len(images) samples for testing
    N_img = len(images)    
    test_indices = np.random.choice(range(0,N_img), int(test_frac*N_img), replace = False)    
    
    test_images = [images[i] for i in test_indices]
    test_classes = [classes[i] for i in test_indices]
    
    train_images = [images[i] for i in range(0,N_img) if i not in test_indices]
    train_classes = [classes[i] for i in range(0,N_img) if i not in test_indices]
    
    return train_images, train_classes, test_images, test_classes