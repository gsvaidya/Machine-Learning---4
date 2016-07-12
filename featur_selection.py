import numpy as np
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
 
# let's read in the yeast data that you used in an earlier assignment:
data = np.genfromtxt("../data/yeast2.csv", delimiter = ",")
X = data[:,1:]
y = data[:,0]
# add some extra features as noise
X = np.hstack((X, np.random.randn(len(y), 250)))
 
# create an instance of RFE that uses an SVM to define weights
# for the features (any linear classifier will work):
classifier = LinearSVC()
selector = RFE(classifier, step=0.1,n_features_to_select=25)
# run feature selection:
selector = selector.fit(X, y)
 
# check which features got chosen:
print selector.support_
print selector.ranking_
 
# to actually perform feature selection:
Xt=selector.fit_transform(X,y)
 
# the wrong way to perform cross-validation:
cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
print np.mean(cross_validation.cross_val_score(classifier, Xt, y, cv=cv))
 
# now let's perform nested cross-validation:
classifier = LinearSVC()
selector = RFE(classifier, step=0.1,n_features_to_select=25)
rfe_svm = make_pipeline(selector, classifier)
 
print np.mean(cross_validation.cross_val_score(rfe_svm, X, y, cv=cv))
 
# feature selection using a univariate filter method:
from sklearn.feature_selection import SelectKBest, f_regression
filter_selector = SelectKBest(f_regression, k=25)
classifier = LinearSVC()
filter_svm = make_pipeline(filter_selector, classifier)
 
print np.mean(cross_validation.cross_val_score(filter_svm, X, y, cv=cv))