from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file
import numpy as np


X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file('leu.t')
y = np.genfromtxt('arcene_train.labels')


def chooseC(X,y): 
    param_grid = [
 {'linearsvc__C': [0.001,001,0.1,1, 10, 100, 1000]},]
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    classifier = LinearSVC()
    selector = RFECV(classifier, step=40)
    rfe_svm = make_pipeline(selector, classifier)
    grid_search = GridSearchCV(rfe_svm, param_grid=param_grid)
    return np.mean(cross_validation.cross_val_score(grid_search, X, y, cv=cv))

def part3(X,y):
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    classifier = LinearSVC(penalty='l2',dual=False)
    selector = RFECV(classifier, step=40)
    rfe_svm = make_pipeline(selector, classifier)
    return np.mean(cross_validation.cross_val_score(rfe_svm, X, y, cv=cv))

if __name__=='__main__' :
    arc1 = chooseC(X2,y)
    arc2 =part3(X2,y)
    print arc1,'arc_nested',arc2 ,'arc'
    leu = chooseC(X_train,y_train)
    leu2 = part3(X_train,y_train)
    print leu,'nest',leu2 ,'leu'