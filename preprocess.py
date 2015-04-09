# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 23:16:22 2015

@author: Hans-Christiaan
"""

import os
import pylab
import mahotas as mh

import skimage.io as imgIO
import skimage.measure as meas

TEST_IMG_PATH = '../train/crustacean_other/87567.jpg'
INPUT_PATH = "../train"
OUTPUT_PATH = "../train_preprocessed"

def showImage(image):
    pylab.gray()
    pylab.imshow(image)
    pylab.show()

def removeNoise(image):
    # blur the image so thresholding works better
    image_blurred = (mh.gaussian_filter(image, 2)).astype('B')
    # threshold the image and label the not connected components    
    labeled, nr_objects = mh.label(image_blurred < image_blurred.mean())
    # compute the properties of the components
    props = meas.regionprops(labeled)
    # get the label of the component with the largest area
    regions = [(prop.area, prop.label) for prop in props]
    label_largest_region = sorted(regions, reverse = True)[0][1]
    
    # set the pixels that are decided to not be the plankton to white
    image_mask = [pixel == label_largest_region for pixel in labeled]
    image_filtered = image * image_mask
    image_filtered[image_filtered == 0] = 255
    
    return image_filtered

def set_progress_bar(progress):
    perc = progress/5
    print '\r[{0}] {1}%'.format('#'*perc+" "*(20-perc), progress),

def preprocessAndSaveImages(input_path, output_path):
    """ Preprocesses the images in 'input_path' and saves them in 'output_path',
        retaining the class directory structure.
        NOTE: the images and directory structure of 'output_path' should already
                exist!
    """        
    class_dirs = os.listdir(input_path)

    img_paths = []    
        
    for class_dir in class_dirs:            
        # get the paths to all of the images in this class
        class_input_path = os.path.join(input_path, class_dir)
        img_names = os.listdir(class_input_path)     
      
        img_paths.extend([os.path.join(class_dir,img_name) for img_name in img_names])        

    total_l = float(len(img_paths))
    print "Amount of images =", total_l    
    print "Preprocessing images..."
    img_counter = 0.0    
    
    for img_path in img_paths:
        img_input_path = os.path.join(input_path, img_path)
        img_output_path = os.path.join(output_path, img_path)        
        
        img = imgIO.imread(img_input_path)
        img = preprocess(img)
        imgIO.imsave(img_output_path, img)
        
        img_counter = img_counter + 1
        set_progress_bar(int(img_counter/total_l*100))
        
# Script for preprocessing the data
def preprocess(image):
    return removeNoise(image)
   
    