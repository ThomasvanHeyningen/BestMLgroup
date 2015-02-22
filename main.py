__author__ = ''

#Import libraries for doing image analysis
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from pylab import cm
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd


#imports van eigen classes
import multiclass_log_loss
import readImages
import featureExtraction

import warnings
warnings.filterwarnings("ignore")

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

    imageReader=readImages.ImageReader(directory_names)
    (images,y,classnames) = imageReader.read()

    featureExtractor=featureExtraction.featureExtractor(imageReader.getMaxPixel(), imageReader.getNumberOfImages())
    X = featureExtractor.extract(images)

    (y_pred, y_pred2) = trainclf(X, y, classnames)
    score=multiclass_log_loss.MulticlassLogLoss()
    print score.calculate_log_loss(y, y_pred2)