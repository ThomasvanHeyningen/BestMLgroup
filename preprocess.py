# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 23:16:22 2015

@author: Hans-Christiaan
"""

import pylab
import mahotas as mh
import math
from skimage.io import imread

TEST_IMG_PATH = '../train/appendicularian_straight/33196.jpg'

def removeNoise(image_path):
    # read image 
    # (testing purposes only, final algorithm will get the image as an argument)
    image = imread(image_path)
    # blur the image so thresholding works better
    image_blurred = (mh.gaussian_filter(image, 2)).astype('B')
    
    # get all the individual connected components and label them
    labeled, nr_objects = mh.label(image_blurred < image_blurred.mean())
    # get the label of the plankton, use the center pixel as a heuristic
    # TODO: use the biggest 'blob' instead
    center_y = math.floor(labeled.shape[0]/2)
    center_x = math.floor(labeled.shape[1]/2)
    plankton_label = labeled[center_y][center_x]    
    # set the pixels that are decided to not be the plankton to white
    image_mask = [pixel == plankton_label for pixel in labeled]
    image_filtered = image * image_mask
    image_filtered[image_filtered == 0] = 255
    # show the before and after image, 
    # for testing purposes only!
    pylab.gray()
    pylab.imshow(image)
    pylab.show()
    pylab.imshow(image_filtered)    
    pylab.show()
    
    return image_filtered

# Script for preprocessing the data
def preprocess(image_path):
    return removeNoise(image_path)
    