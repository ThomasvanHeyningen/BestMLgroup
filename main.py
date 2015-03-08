__author__ = ''

#Import libraries for doing image analysis
import os

import numpy as np

#imports van eigen classes
import multiclass_log_loss
import readImages
import featureExtraction
import prediction
import ConfigFileReader

import warnings
warnings.filterwarnings("ignore")

#Voorbeeld voor config!
#C = ConfigFileReader.ConfigFileReader()
#print (C.getVariable("ClassicClassifier", "name"))

if __name__ == '__main__':
    C = ConfigFileReader.ConfigFileReader()
    # Training data is located according to configFile.ini
    train_dir = os.path.join(os.getcwd(), C.getVariable("Directories", "train"))
    # get the classnames from the directory structure    
    directory_names = [os.path.join(train_dir,class_name)\
        for class_name in os.listdir(train_dir)]
    
    imageReader=readImages.ImageReader(directory_names)
    (images,y,classnames) = imageReader.read()

    print "extracting features"
    featureExtractor=featureExtraction.featureExtractor(imageReader.getMaxPixel(), imageReader.getNumberOfImages())
    X  = featureExtractor.extract(images)
    X2 = featureExtractor.getCNNfeatures(imageReader.files)
    X = np.hstack([X,X2])

    print "training classifier"
    predictor=prediction.predictor(5,100) # n_folds, n_estimators
    (y_pred, y_prob) = predictor.trainclf(X, y, classnames)
    print "calculating scores"
    score=multiclass_log_loss.MulticlassLogLoss()
    print score.calculate_log_loss(y, y_prob)
