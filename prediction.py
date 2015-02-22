from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
import numpy as np

class featureExtractor():

    def __init__(self, maxPixel, numberOfImages):
        self.maxPixel=maxPixel
        self.imageSize=maxPixel*maxPixel
        self.numberOfImages=numberOfImages

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