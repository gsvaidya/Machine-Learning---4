#RFESVM part 3

import numpy as np
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_svmlight_file

X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file('leu.t')
y = np.genfromtxt('arcene_train.labels')
result = []
result2 = []

def chooseC(X,y): 
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
        param_grid = [
    {'linearsvc__C': [0.001,001,0.1,1, 10, 100, 1000]},]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        classifier = LinearSVC()
        selector = RFE(classifier, step=0.1,n_features_to_select=int(j[i]*X.shape[1]))
        rfe_svm = make_pipeline(selector, classifier)
        grid_search = GridSearchCV(rfe_svm, param_grid=param_grid)
        result.append(np.mean(cross_validation.cross_val_score(grid_search, X, y, cv=cv)))
    return result
    
def part3(X,y):
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        classifier = LinearSVC(penalty='l2',dual=False)
        selector = RFE(classifier, step=0.1,n_features_to_select=int(j[i]*X.shape[1]))
        rfe_svm = make_pipeline(selector, classifier)
        result2.append(np.mean(cross_validation.cross_val_score(rfe_svm, X, y, cv=cv)))
    return result2

if __name__=='__main__' :
    arc1 = chooseC(X2,y)
    arc2 =part3(X2,y)
    print arc1,'arc_nested',arc2 ,'arc'
    leu = chooseC(X_train,y_train)
    leu2 = part3(X_train,y_train)
    print leu,'nest',leu2 ,'leu'