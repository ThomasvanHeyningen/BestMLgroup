from skimage import measure
from skimage import morphology
import numpy as np
import csv
import os
import ConfigFileReader
import math
from skimage.segmentation._quickshift import ndimage
from astropy.coordinates.earth_orientation import eccentricity

class featureExtractor():

    def __init__(self, maxPixel, numberOfImages):
        self.maxPixel=maxPixel
        self.imageSize=maxPixel*maxPixel
        self.numberOfImages=numberOfImages

    def getLargestRegion(self, props, labelmap, imagethres):
    # find the largest nonzero region
        regionmaxprop = None
        for regionprop in props:
            # check to see if the region is at least 50% nonzero
            if sum(imagethres[labelmap == regionprop.label])*1.0/regionprop.area < 0.50:
                continue
            if regionmaxprop is None:
                regionmaxprop = regionprop
            if regionmaxprop.filled_area < regionprop.filled_area:
                regionmaxprop = regionprop
        return regionmaxprop

    def getMinorMajorRatio(self, image):
        image = image.copy()
        # Create the thresholded image to eliminate some of the background
        imagethr = np.where(image > np.mean(image),0.,1.0)

        #Dilate the image
        imdilated = morphology.dilation(imagethr, np.ones((4,4)))
        # Create the label list
        label_list = measure.label(imdilated)
        label_list = imagethr*label_list
        label_list = label_list.astype(int)

        region_list = measure.regionprops(label_list)
        maxregion = self.getLargestRegion(region_list, label_list, imagethr)
        
        # guard against cases where the segmentation fails by providing zeros
        ratio = 0.0
        width = 0.0
        height = 0.0
        centroidrow = 0.0
        centroidcol = 0.0
        convex_area = 0.0
        area= 0.0
        perimeter = 0.0
        euler = 0.0
        circularity = 0.0
        solidity = 0.0
        eccentricity = 0.0
        rectangularity = 0.0
        if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
            ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
            width = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0
            height = 0.0 if maxregion is None else  1.0*maxregion.major_axis_length
            (centroidrow,centroidcol) = (0.0,0.0) if maxregion is None else maxregion.centroid
            centroidrow=centroidrow*1.0
            centroidcol=centroidcol*1.0
            convex_area = 0.0 if maxregion is None else  1.0*maxregion.convex_area
            area = 0.0 if maxregion is None else  1.0*maxregion.area
            perimeter = 0.0 if maxregion is None else  1.0*maxregion.perimeter
            euler = 0.0 if maxregion is None else  1.0*maxregion.euler_number
            if (maxregion.perimeter != 0.0):
                circularity = (math.pi*4*area)/(perimeter*perimeter)
            solidity = 0.0 if maxregion is None else area / convex_area
            eccentricity = 0.0 if maxregion is None else 1.0*maxregion.eccentricity
            rectangularity = 0.0 if maxregion is None else 1.0*maxregion.extent
        return ratio, width, height, area, centroidrow, centroidcol, convex_area, perimeter, circularity, euler, solidity, eccentricity, rectangularity

    def getBwRatio(self, image):
        image = image.copy()
        bwmean = sum(image) / len(image)
        '''
        #Other possible solution disregarding the white background of the image
        numberOfOnes = (image == 1).sum()
        bwmean = (sum(image)-numberOfOnes) / (len(image)-numberOfOnes)
        '''
        return bwmean

    def edges(self, image):
        sx = ndimage.sobel(image, axis=0, mode='constant')
        sy = ndimage.sobel(image, axis=1, mode='constant')
        sob = np.hypot(sx,sy)
        return sob

    def extract(self, images, addImage):
        numberOfFeatures=14
        if not addImage:
            self.imageSize=0
        X= np.zeros((self.numberOfImages, self.imageSize+numberOfFeatures + 625), dtype=float)

        for i in range(0,self.numberOfImages):
            if addImage:
                X[i, 0:self.imageSize] = images[i]
            image=np.reshape(images[i], (self.maxPixel, self.maxPixel))
            (axisratio, width, height, area, centroidrow, centroidcol, convex_area, perimeter, circularity, euler, solidity, eccentricity, rectangularity) = self.getMinorMajorRatio(image)
            sob = self.edges(image)
            r, c = sob.shape
            sob = np.reshape(sob, (r*c), 1)
            image=np.reshape(images[i], (self.maxPixel*self.maxPixel, 1))
            bwmean = self.getBwRatio(image)
            #(newfeature) = function(image)
            X[i, self.imageSize+0] = axisratio
            X[i, self.imageSize+1] = height # this might not be good
            X[i, self.imageSize+2] = width# this might not be
            X[i, self.imageSize+3] = centroidrow
            X[i, self.imageSize+4] = centroidcol
            X[i, self.imageSize+5] = convex_area
            X[i, self.imageSize+6] = perimeter
            X[i, self.imageSize+7] = circularity
            X[i, self.imageSize+8] = euler
            X[i, self.imageSize+9] = area
            X[i, self.imageSize+10] = bwmean
            X[i, self.imageSize+11] = solidity
            X[i, self.imageSize+12] = eccentricity
            X[i, self.imageSize+13] = rectangularity
            X[i, (self.imageSize+numberOfFeatures):(self.imageSize+numberOfFeatures+len(sob))] = sob
            #X[i, self.imageSize+3] = newfeature
        return X

    def getCNNfeatures(self, names, which):
        C = ConfigFileReader.ConfigFileReader()
        f_dir  = C.getVariable('Directories', 'CNNdir')
        if which == 'train':
            f_file = C.getVariable('Directories', 'CNNtrainfeatures')
        else:
            f_file = C.getVariable('Directories', 'CNNtestfeatures')
        feat_reader = csv.reader(open(os.path.join(f_dir, f_file)), delimiter=',')
        
        feat_dict = {}
        q = 0
        for line in feat_reader:
            q+=1
            lbl  = line[0]
            name = line[1]
            if name in feat_dict: print(q)
            feat_dict.update({name: line[2:]})
            #lbl_dict.update({name: lbl})
        print(q)
        nfeatures = len(feat_dict.itervalues().next())
        features = np.zeros((len(names), nfeatures))
        #labels   = np.zeros( len(names))
        print("nfeatures: ", nfeatures, ", nImgs: ", len(names))
        for i, name in enumerate(names):
            name = os.path.split(name)[1]
            features[i] = feat_dict[name]
              #labels[i] =  lbl_dict[name]
        return features#, labels # Actually don't need labels
