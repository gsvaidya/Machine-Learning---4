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
for i in range(10):
    classifier = LinearSVC(penalty='l1',dual = False)
    classifier = classifier.fit(X2,y)
    Xt = classifier.fit_transform(X2,y)
    nz = classifier.coef_
    nz2 = np.transpose(nz)
    count[i] = np.count_nonzero(nz2)
print np.mean(count)
#ind = np.nonzero(nz2)
#l = list(ind[0])
#new = []
#y2 = []
#for i in range(len(l)):
#    j = l[i]
#    new.append(nz2[j])



   



#selector = RFE(classifier, step=0.1,n_features_to_select=25)
#rfe_svm = make_pipeline(selector, classifier)
#selector = selector.fit(X2,y)
#
#print np.mean(cross_validation.cross_val_score(rfe_svm, X2, y, cv=5))