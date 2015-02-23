__author__ = ''

#Import libraries for doing image analysis
import glob
import os


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
    # get the classnames from the directory structure
    directory_names = list(set(glob.glob(os.path.join("..","train", "*"))\
    ).difference(set(glob.glob(os.path.join("..","train","*.*")))))

    imageReader=readImages.ImageReader(directory_names)
    (images,y,classnames) = imageReader.read()

    featureExtractor=featureExtraction.featureExtractor(imageReader.getMaxPixel(), imageReader.getNumberOfImages())
    X = featureExtractor.extract(images)

    predictor=prediction.predictor(5,100) # n_folds, n_estimators
    (y_pred, y_prob) = predictor.trainclf(X, y, classnames)

    score=multiclass_log_loss.MulticlassLogLoss()
    print score.calculate_log_loss(y, y_prob)


