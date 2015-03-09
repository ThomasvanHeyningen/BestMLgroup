from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF

import numpy as np


class predictor():

    def __init__(self, n_folds, n_estimators):
        self.n_folds=n_folds
        self.n_estimators=n_estimators

    def trainfoldedclf(self, X, y, namesClasses):
        kf = KFold(y, self.n_folds)
        y_pred = y * 0
        y_prob = np.zeros((len(y),len(set(y))))
        for train, test in kf:
            print "running fold"
            X_train, X_test, y_train, y_test = X[train,:], X[test,:], y[train], y[test]
            clf = RF(self.n_estimators, n_jobs=3)
            clf.fit(X_train, y_train)
            y_pred[test] = clf.predict(X_test)
            y_prob[test] = clf.predict_proba(X_test)
        print classification_report(y, y_pred, target_names=namesClasses)
        #scores = cross_validation.cross_val_score(clf, X, y, cv=self.n_folds, n_jobs=1);
        #print "Accuracy of all classes"
        #print np.mean(scores)
        return(y_pred, y_prob, clf)

    def trainunfoldedclf(self, X, y):
        clf = RF(self.n_estimators, n_jobs=3)
        clf.fit(X, y)
        return(clf)

