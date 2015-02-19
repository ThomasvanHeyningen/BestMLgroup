__author__ = ''

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
import multiclass_log_loss
from scipy import ndimage
from skimage.feature import peak_local_max

import warnings
warnings.filterwarnings("ignore")

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
        #there's a chance that width and height are the wrong way around.
    return ratio, width, height

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
    kf = KFold(y, n_folds=5) #stond op 5
    y_pred = y * 0
    y_pred2 = np.zeros((len(y),len(set(y))))
    for train, test in kf:
        print "running fold"
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
        clf = RF(n_estimators=100, n_jobs=3) #stond op 100
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
        y_pred2[test] = clf.predict_proba(X_test)
    print classification_report(y, y_pred, target_names=namesClasses)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, n_jobs=1); #stond op 5
    print "Accuracy of all classes"
    print np.mean(scores)
    return(y_pred, y_pred2)

if __name__ == '__main__':
    # get the classnames from the directory structure
    directory_names = list(set(glob.glob(os.path.join("..","train", "*"))\
    ).difference(set(glob.glob(os.path.join("..","train","*.*")))))

    (X,y,classnames,num_features) = readimages()
    (y_pred, y_pred2) = trainclf(X, y, classnames)
    score=multiclass_log_loss.MulticlassLogLoss()
    print score.calculate_log_loss(y, y_pred2)