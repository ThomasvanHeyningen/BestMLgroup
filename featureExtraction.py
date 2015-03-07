from skimage import measure
from skimage import morphology
import numpy as np
import math

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
        return ratio, width, height, area, centroidrow, centroidcol, convex_area, perimeter, circularity, euler

    def getBwRatio(self, image):
        image = image.copy()
        bwmean = sum(image) / len(image)
        '''
        #Other possible solution disregarding the white background of the image
        numberOfOnes = (image == 1).sum()
        bwmean = (sum(image)-numberOfOnes) / (len(image)-numberOfOnes)
        '''
        return bwmean

    def extract(self, images, addImage):
        numberOfFeatures=11
        if not addImage:
            self.imageSize=0
        X= np.zeros((self.numberOfImages, self.imageSize+numberOfFeatures), dtype=float)

        for i in range(0,self.numberOfImages):
            if addImage:
                X[i, 0:self.imageSize] = images[i]
            image=np.reshape(images[i], (self.maxPixel, self.maxPixel))
            (axisratio, width, height, area, centroidrow, centroidcol, convex_area, perimeter, circularity, euler) = self.getMinorMajorRatio(image)
            image=np.reshape(images[i], (self.maxPixel*self.maxPixel, 1))
            bwmean = self.getBwRatio(image)
            #(newfeature) = function(image)
            #X[i, self.imageSize+3] = newfeature
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
        return X