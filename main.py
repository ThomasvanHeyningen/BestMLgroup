__author__ = ''

#Import libraries for doing image analysis
import glob
import os
import time


#imports van eigen classes
import multiclass_log_loss
import readImages
import featureExtraction
import prediction
import submission

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # get the classnames from the directory structure
    debug=False
    test=False

    directory_names = list(set(glob.glob(os.path.join("..","train", "*"))\
    ).difference(set(glob.glob(os.path.join("..","train","*.*")))))
    test_directory=glob.glob(os.path.join("..", "test"))
    if debug:
        directory_names= list(set(glob.glob(os.path.join("..","smalltrainset", "*"))\
        ).difference(set(glob.glob(os.path.join("..","smalltrainset","*.*")))))
        test_directory=glob.glob(os.path.join("..", "smalltestset"))

    imageReader=readImages.ImageReader(directory_names)
    (images,y,classnames) = imageReader.read()

    print "extracting features"
    featureExtractor=featureExtraction.featureExtractor(imageReader.getMaxPixel(), imageReader.getNumberOfImages())
    X = featureExtractor.extract(images)

    print "training classifier"
    predictor=prediction.predictor(5,100) # n_folds, n_estimators
    (y_pred, y_prob, clf) = predictor.trainclf(X, y, classnames)

    if not test:
        print "calculating scores"
        score=multiclass_log_loss.MulticlassLogLoss()
        print score.calculate_log_loss(y, y_prob)

    if test:
        print "loading images for test set"
        testImageReader=readImages.ImageReader(test_directory)
        (testimages, imagefilenames) = testImageReader.readtest()

        print "extracting features for test set"
        featureExtractor=featureExtraction.featureExtractor(testImageReader.getMaxPixel(), testImageReader.getNumberOfImages())
        testset = featureExtractor.extract(testimages)

        imageTester=submission.Tester(clf, classnames)
        imageTester.test(testset, imagefilenames)

