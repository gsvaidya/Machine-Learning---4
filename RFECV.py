import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC

X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file('leu.t')
y = np.genfromtxt('arcene_train.labels')

def RFE_CV(X,y):
    classifier = LinearSVC(penalty='l2',dual=False)
    classifier.fit(X,y)
    filter_selector = RFECV(classifier, step =0.01 )
    filter_svm = make_pipeline(filter_selector, classifier)
    
    return np.mean(cross_validation.cross_val_score(filter_svm, X, y, cv=5))

if __name__=='__main__' :
    print RFE_CV(X2,y),'arcene'
    print RFE_CV(X_train,y_train),'leukemia'