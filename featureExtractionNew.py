# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 17:40:45 2015

@author: Hans-Christiaan
"""

import skimage.measure as meas
import skimage.io as imgIO
import mahotas as mh

TEST_IMG_PATH = '../train/crustacean_other/87567.jpg'

def getLargestRegion(props):
    # get the properties of the component with the largest area
    regions = [(prop.area, prop) for prop in props]
    largest_prop = sorted(regions, reverse = True)[0][1]
    return largest_prop
    
def getRegionFeatures(image):
    # blur the image so thresholding works better
    image_blurred = (mh.gaussian_filter(image, 2)).astype('B')
    # threshold the image and label the components    
    labeled, nr_objects = mh.label(image_blurred < image_blurred.mean())
    # compute the properties of the components
    props = meas.regionprops(labeled, cache = False)
    # get the properties of the largest component
    # (assumed to be the plankton)
    l_prop = getLargestRegion(props)
    # for a description of the properties, see:
    # http://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops
    return [l_prop.area, l_prop.eccentricity, l_prop.equivalent_diameter, \
            l_prop.major_axis_length, l_prop.minor_axis_length, l_prop.orientation]
            
def loadImage(image_path):
    return imgIO.imread(image_path)