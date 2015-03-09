__author__ = ''

#Import libraries for doing image analysis
import os
import time

import numpy as np
import glob

#imports van eigen classes
import multiclass_log_loss
import readImages
import featureExtraction
import prediction
import ConfigFileReader
import submission

import warnings
warnings.filterwarnings("ignore")

#Voorbeeld voor config!
#C = ConfigFileReader.ConfigFileReader()
#print (C.getVariable("ClassicClassifier", "name"))
if __name__ == '__main__':
    start_time = time.time()
    C = ConfigFileReader.ConfigFileReader()
    # get the classnames from the directory structure
    debug = False
    test  = False
    n_estimators=300 # make this higher to improve score (and computing time)
    addImage=True # adds the image pixels as features.

    if debug: 
        train_dir = C.getVariable("Directories", "train_small")
        test_dir  = C.getVariable("Directories", "test_small")
    else:
        train_dir = C.getVariable("Directories", "train")
        test_dir  = C.getVariable("Directories", "test")
    directory_names = list(set(glob.glob(os.path.join(train_dir, "*"))\
    ).difference(set(glob.glob(os.path.join(train_dir, "*.*")))))
    test_directory=glob.glob(test_dir)

    imageReader=readImages.ImageReader(directory_names)
    (images,y,classnames, namesfortest) = imageReader.read()
    featureExtractor = featureExtraction.featureExtractor(imageReader.getMaxPixel(), imageReader.getNumberOfImages())

    X  = featureExtractor.extract(images, None)
    X2 = featureExtractor.getCNNfeatures(imageReader.files, 'train')
    X  = np.hstack([X,X2])
    if not test:
        print "training classifier with folds"
        predictor=prediction.predictor(5,n_estimators) # n_folds, n_estimators
        (y_pred, y_prob, clf) = predictor.trainfoldedclf(X, y, classnames)

        print "calculating scores"
        score=multiclass_log_loss.MulticlassLogLoss()
        print score.calculate_log_loss(y, y_prob)

    if test:
        print "training classifier"
        predictor=prediction.predictor(5,n_estimators) # n_folds, n_estimators
        (clf) = predictor.trainunfoldedclf(X, y)

        print "loading images for test set"
        testImageReader=readImages.ImageReader(test_directory)
        (testimages, imagefilenames) = testImageReader.readtest()

        print "extracting features for test set"
        featureExtractor=featureExtraction.featureExtractor(testImageReader.getMaxPixel(), testImageReader.getNumberOfImages())
        testset = featureExtractor.extract(testimages, addImage)
        testset2= featureExtractor.extractCNNfeatures(testImageReader.files, 'test')

        testset = np.hstack([testset,testset2])

        imageTester=submission.Tester(clf, namesfortest)
        imageTester.test(testset, imagefilenames)
    print("--- execution took %s seconds ---" % (time.time() - start_time))

