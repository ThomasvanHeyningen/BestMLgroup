from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
import ConfigFileReader
import re

class ImageReader:

    def __init__(self, directory_names):
        self.directory_names=directory_names
        self.numberofImages = 0
        self.maxPixel = 25
        self.C = ConfigFileReader.ConfigFileReader()

    def deriveNumberOfImages(self):
        #loops through the training categories to find the number of images
        for folder in self.directory_names:
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                     # Only read in the images
                    if fileName[-4:] == ".jpg":
                      self.numberofImages += 1

    def setMaxPixel(self, pixelSize):
        self.maxPixel=pixelSize

    def getMaxPixel(self):
        return self.maxPixel

    def getImageSize(self):
        return self.maxPixel*self.maxPixel

    def setNumberOfImages(self, numberOfImages):
        self.numberofImages=numberOfImages

    def getNumberOfImages(self):
        return self.numberofImages

    def read(self):

        #get the number of images first, we can later change this to allow the number to be given for quicker running.
        self.deriveNumberOfImages()

        #Rescale the images and create the combined metrics and training labels
        imageSize = self.maxPixel * self.maxPixel
        num_rows = self.numberofImages # one row for each image in the training dataset

        # X is the feature vector with one row of features per image
        # consisting of the pixel values and our metric
        X = np.zeros((num_rows, imageSize), dtype=float)
        # y is the numeric class label
        y = np.zeros((num_rows))

        self.files = []
        # Generate training data
        i = 0
        label = 0
        # List of string of class names
        namesClasses = list()
        classnames = list()

        print "Reading images"
        # Navigate through the list of directories
        for folder in self.directory_names:
            # Append the string class name for each class
            currentClass = folder.split(os.pathsep)[-1]
            className=re.sub('\.\.\\\\train\\\\', "", currentClass)
            classnames.append(className)
            namesClasses.append(currentClass)
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                    # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue

                    # Read in the images and create the features
                    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                    image = imread(nameFileImage, as_grey=True)
                    self.files.append(nameFileImage)
                    image = resize(image, (self.maxPixel, self.maxPixel))

                    # Store the rescaled image pixels and the axis ratio
                    X[i, 0:imageSize] = np.reshape(image, (1, imageSize))

                    # Store the classlabel
                    y[i] = label
                    i += 1
                    # report progress for each 5% done
                    report = [int((j+1)*num_rows/20.) for j in range(20)]
                    if i in report:
                        print np.ceil(i *100.0 / num_rows), "% done"
            label += 1
        return (X,y,namesClasses,classnames)

    def readtest(self):

        #get the number of images first, we can later change this to allow the number to be given for quicker running.
        self.deriveNumberOfImages()

        #Rescale the images and create the combined metrics and training labels
        imageSize = self.maxPixel * self.maxPixel
        num_rows = self.numberofImages # one row for each image in the training dataset

        # X is the feature vector with one row of features per image
        # consisting of the pixel values and our metric
        X = np.zeros((num_rows, imageSize), dtype=float)
        # y is the numeric class label
        y = np.zeros((num_rows))

        files = []
        # Generate training data
        i = 0
        label = 0
        # List of string of class names
        namesClasses = list()
        namesFiles = list()

        print "Reading testset images"
        # Navigate through the list of directories
        for folder in self.directory_names:
            # Append the string class name for each class
            currentClass = folder.split(os.pathsep)[-1]
            namesClasses.append(currentClass)
            for fileNameDir in os.walk(folder):
                for fileName in fileNameDir[2]:
                    # Only read in the images
                    if fileName[-4:] != ".jpg":
                      continue

                    # Read in the images and create the features
                    nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                    imageName=re.sub('\.\.\\\\test\\\\test\\\\', "", nameFileImage)
                    namesFiles.append(imageName)
                    image = imread(nameFileImage, as_grey=True)
                    files.append(nameFileImage)
                    image = resize(image, (self.maxPixel, self.maxPixel))

                    # Store the rescaled image pixels and the axis ratio
                    X[i, 0:imageSize] = np.reshape(image, (1, imageSize))

                    # Store the classlabel
                    y[i] = label
                    i += 1
                    # report progress for each 5% done
                    report = [int((j+1)*num_rows/20.) for j in range(20)]
                    if i in report: print np.ceil(i *100.0 / num_rows), "% done with test set"
            label += 1
        return (X, namesFiles)
