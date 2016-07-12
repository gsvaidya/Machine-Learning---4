import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC

X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file("leu.t")
y = np.genfromtxt('arcene_train.labels')

def L1(X,y):
    classifier = LinearSVC(penalty ='l1',dual=False)
    classifier.fit(X,y)  
    #cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    return np.mean(cross_validation.cross_val_score(classifier, X, y, cv=5))

def L2(X,y):
    classifier2 = LinearSVC(penalty ='l2',dual=False)   
    classifier2.fit(X,y)
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    return np.mean(cross_validation.cross_val_score(classifier2, X, y, cv=cv))
    
if __name__=='__main__' :
    arc_l1 = L1(X2,y)
    arc_l2 = L2(X2,y)
    print arc_l1,'arc_l1',arc_l2,'arc'
    
    leu_l1 = L1(X_train,y_train)
    leu_l2 = L2(X_train,y_train)
    print leu_l1,'leu_l1',leu_l2,'leu_l2'