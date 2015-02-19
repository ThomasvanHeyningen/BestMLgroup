__author__ = 'Plankton'

#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max

import warnings
warnings.filterwarnings("ignore")

# get the classnames from the directory structure
directory_names = list(set(glob.glob(os.path.join("..","train", "*"))\
 ).difference(set(glob.glob(os.path.join("..","train","*.*")))))

def exampleimage():
    # Example image
    # This example was chosen for because it has two noncontinguous pieces
    # that will make the segmentation example more illustrative
    example_file = glob.glob(os.path.join(directory_names[5], "*.jpg"))[9]
    print example_file
    im = imread(example_file, as_grey=True)
    plt.imshow(im, cmap=cm.gray)
    plt.show()

    # First we threshold the image by only taking values greater than the mean to reduce noise in the image
    # to use later as a mask
    f = plt.figure(figsize=(12,3))
    imthr = im.copy()
    imthr = np.where(im > np.mean(im),0.,1.0)
    sub1 = plt.subplot(1,4,1)
    plt.imshow(im, cmap=cm.gray)
    sub1.set_title("Original Image")

    sub2 = plt.subplot(1,4,2)
    plt.imshow(imthr, cmap=cm.gray_r)
    sub2.set_title("Thresholded Image")

    imdilated = morphology.dilation(imthr, np.ones((4,4)))
    sub3 = plt.subplot(1, 4, 3)
    plt.imshow(imdilated, cmap=cm.gray_r)
    sub3.set_title("Dilated Image")

    labels = measure.label(imdilated)
    labels = imthr*labels
    labels = labels.astype(int)
    sub4 = plt.subplot(1, 4, 4)
    sub4.set_title("Labeled Image")
    plt.imshow(labels)

    # calculate common region properties for each region within the segmentation
    regions = measure.regionprops(labels)

# find the largest nonzero region
def getLargestRegion(props, labelmap, imagethres):
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

def getMinorMajorRatio(image):
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
    maxregion = getLargestRegion(region_list, label_list, imagethr)

    # guard against cases where the segmentation fails by providing zeros
    ratio = 0.0
    width = 0.0
    height = 0.0
    if ((not maxregion is None) and  (maxregion.major_axis_length != 0.0)):
        ratio = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0 / maxregion.major_axis_length
        width = 0.0 if maxregion is None else  maxregion.minor_axis_length*1.0
        height = 0.0 if maxregion is None else  1.0*maxregion.major_axis_length
    return ratio, width, height

def readimages():
    #Rescale the images and create the combined metrics and training labels
    #get the total training images
    numberofImages = 0
    for folder in directory_names:
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                 # Only read in the images
                if fileName[-4:] != ".jpg":
                  continue
                numberofImages += 1

    # We'll rescale the images to be 25x25
    maxPixel = 25
    imageSize = maxPixel * maxPixel
    num_rows = numberofImages # one row for each image in the training dataset
    num_features = imageSize + 1 # for our ratio

    # X is the feature vector with one row of features per image
    # consisting of the pixel values and our metric
    X = np.zeros((num_rows, num_features), dtype=float)
    # y is the numeric class label
    y = np.zeros((num_rows))

    files = []
    # Generate training data
    i = 0
    label = 0
    # List of string of class names
    namesClasses = list()

    print "Reading images"
    # Navigate through the list of directories
    for folder in directory_names:
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
                image = imread(nameFileImage, as_grey=True)
                files.append(nameFileImage)
                (axisratio, width, height) = getMinorMajorRatio(image)
                image = resize(image, (maxPixel, maxPixel))

                # Store the rescaled image pixels and the axis ratio
                X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
                X[i, imageSize+0] = axisratio
                X[i, imageSize+1] = height # this might not be good
                X[i, imageSize+2] = width# this might not be good

                # Store the classlabel
                y[i] = label
                i += 1
                # report progress for each 5% done
                report = [int((j+1)*num_rows/20.) for j in range(20)]
                if i in report: print np.ceil(i *100.0 / num_rows), "% done"
        label += 1
    return (X,y,namesClasses, num_features)

def makeplots(X, y, namesClasses, num_features):
    # Loop through the classes two at a time and compare their distributions of the Width/Length Ratio

    #Create a DataFrame object to make subsetting the data on the class
    df = pd.DataFrame({"class": y[:], "ratio": X[:, num_features-1]})

    f = plt.figure(figsize=(30, 20))
    #we suppress zeros and choose a few large classes to better highlight the distributions.
    df = df.loc[df["ratio"] > 0]
    minimumSize = 20
    counts = df["class"].value_counts()
    largeclasses = [int(x) for x in list(counts.loc[counts > minimumSize].index)]
    # Loop through 40 of the classes
    for j in range(0,40,2):
        subfig = plt.subplot(4, 5, j/2 +1)
        # Plot the normalized histograms for two classes
        classind1 = largeclasses[j]
        classind2 = largeclasses[j+1]
        n, bins,p = plt.hist(df.loc[df["class"] == classind1]["ratio"].values,\
                             alpha=0.5, bins=[x*0.01 for x in range(100)], \
                             label=namesClasses[classind1].split(os.sep)[-1], normed=1)

        n2, bins,p = plt.hist(df.loc[df["class"] == (classind2)]["ratio"].values,\
                              alpha=0.5, bins=bins, label=namesClasses[classind2].split(os.sep)[-1],normed=1)
        subfig.set_ylim([0.,10.])
        plt.legend(loc='upper right')
        plt.xlabel("Width/Length Ratio")

def trainclf(X, y, namesClasses):
    kf = KFold(y, n_folds=3) #stond op 5
    y_pred = y * 0
    y_pred2 = np.zeros((len(y),len(set(y))))
    for train, test in kf:
        print "one in the forloop"
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
        clf = RF(n_estimators=5, n_jobs=3) #stond op 100
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
        y_pred2[test] = clf.predict_proba(X_test)
    print classification_report(y, y_pred, target_names=namesClasses)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1); #stond op 5
    print "Accuracy of all classes"
    print np.mean(scores)
    return(y_pred, y_pred2)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss



#maincode
(X,y,classnames,num_features) = readimages()
(y_pred, y_pred2) = trainclf(X, y, classnames)

print multiclass_log_loss(y, y_pred2)
