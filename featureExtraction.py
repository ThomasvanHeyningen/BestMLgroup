from skimage import measure
from skimage import morphology
import numpy as np

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
        if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
            ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
            width = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0
            height = 0.0 if maxregion is None else  1.0*maxregion.major_axis_length
            #there's a chance that width and height are the wrong way around.
        return ratio, width, height


    def extract(self, images):
        numberOfFeatures=3
        X= np.zeros((self.numberOfImages, self.imageSize+numberOfFeatures), dtype=float)

        for i in range(0,self.numberOfImages):
            X[i, 0:self.imageSize] = images[i]
            image=np.reshape(images[i], (self.maxPixel, self.maxPixel))
            (axisratio, width, height) = self.getMinorMajorRatio(image)
            X[i, self.imageSize+0] = axisratio
            X[i, self.imageSize+1] = height # this might not be good
            X[i, self.imageSize+2] = width# this might not be good