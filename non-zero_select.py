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
count = np.zeros(10)

def nonzero(X,y):
    for i in range(10):
        classifier = LinearSVC(penalty='l1',dual = False)
        classifier.fit(X,y)
        #Xt = classifier.fit_transform(X2,y)
        nz = classifier.coef_
        nz2 = np.transpose(nz)
        count[i] = np.count_nonzero(nz2)
    return np.mean(count)

def accuracy(X,y):
    cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
    classifier2 = LinearSVC(penalty='l2',dual=False)
    selector = LinearSVC(penalty ='l1',dual=False)
    l1_l2= make_pipeline(selector, classifier2)
    return np.mean(cross_validation.cross_val_score(l1_l2, X, y, cv=cv))

if __name__=='__main__' :
    count_arcene = nonzero(X2,y)
    acc_arcene = accuracy(X2,y)
    print count_arcene, acc_arcene ,'arcene'
    
    count_leukemia = nonzero(X_train,y_train)
    acc_leukemia = accuracy(X_train,y_train)
    print count_leukemia, acc_leukemia ,'leu'
#ind = np.nonzero(nz2)
#l = list(ind[0])
#new = []
#y2 = []
#for i in range(len(l)):
#    j = l[i]
#    new.append(nz2[j])
#print new
