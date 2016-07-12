import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
from sklearn.utils import shuffle

X_train, y_train = load_svmlight_file("leu.t")
#print X_train , y_train,'old'
#X_train , y_train = shuffle (X_train,y_train,random_state=0)
#print X_train , y_train,'new'


#index = np.arange(np.shape(X_train)[0])
#print index
#np.random.shuffle(index)
#print X_train[index, :]