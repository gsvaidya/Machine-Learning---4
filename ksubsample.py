#k-subsample method

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV


X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file("leu.t")
y = np.genfromtxt('arcene_train.labels')
count = np.zeros(30)
#np.set_printoptions(threshold='nan')

def ksubsample(X,y):
    i = 0.8*X.shape[0]
    j = 0.8*y.shape[0]
    i = int(i)
    j = int(j)
    score = np.zeros(X.shape[1])
    for k in range(100):
        X , y = shuffle (X,y,random_state=0)
        X = X[:i]
        y = y[:j]
        classifier = LinearSVC(penalty='l1',dual=False)
        classifier.fit(X,y)
        nz = classifier.coef_
        nz2 = np.transpose(nz)
        ind = np.nonzero(nz2)
        l = list(ind[0])
        for k in range(X.shape[1]):
            for m in range(len(l)):
                if l[m] == k:
                    score[k]=score[k]+1
    return score,score
result =[]
result2 = []
def chooseC(X,y): 
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        param_grid = [
    {'linearsvc__C': [0.01,0.1,1,10,100]},]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        classifier = LinearSVC(penalty='l2',dual=False)
        filter_selector = SelectKBest(ksubsample,k=int(X.shape[1]*j[i]))
        filter_svm = make_pipeline(filter_selector,classifier)
        grid_search = GridSearchCV(filter_svm, param_grid=param_grid)
        result.append (np.mean(cross_validation.cross_val_score(grid_search, X, y, cv=cv)))
        
    return result

def part3(X,y):
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        filter_selector = SelectKBest(ksubsample,k=int(X.shape[1]*j[i]))
        classifier = LinearSVC(penalty='l2',dual=False)
        filter_svm = make_pipeline(filter_selector, classifier)
        result.append (np.mean(cross_validation.cross_val_score(filter_svm, X, y, cv=cv)))
        

if __name__=='__main__' :
    #arcene = ksubsample(X2,y)
    arcene_2 = chooseC(X2,y)
    arcene_3 = part3(X2,y)
    print arcene_2,'nested',arcene_3,'general'
    
    #leukemia = ksubsample(X_train,y_train)
    leukemia_2 = chooseC(X_train,y_train)
    leukemia_3 = part3(X_train,y_train)
    print leukemia_2,'nested',leukemia_3,'leu'
    