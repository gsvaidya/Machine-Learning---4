import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn import svm, feature_selection, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE,RFECV
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.grid_search import GridSearchCV


X1 = np.genfromtxt('arcene_test.data')
X2 = np.genfromtxt('arcene_train.data')
X_train, y_train = load_svmlight_file("leu.t")
y = np.genfromtxt('arcene_train.labels')

np.set_printoptions(threshold='nan')

def golub(X,y):
    neg = []
    pos = []
    result = []
    for i in range((X.shape[1])):
        for j in range(len(y)):
            if y[j] < 0:
                neg.append(X[j,i])
            else:
                pos.append(X[j,i])
        p_mean = np.mean(pos)
        n_mean = np.mean(neg)
        p_sd = np.std(pos)
        n_sd = np.std(neg)
        Num = abs(p_mean - n_mean)
        Den = p_sd + n_sd
        result.append(Num/Den)
    return result,result
result = []
result2 = []
def chooseC(X,y): 
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        param_grid = [
    {'linearsvc__C': [0.01,0.1,1,10,100]},]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        classifier = LinearSVC(penalty='l2',dual=False)
        filter_selector = SelectKBest(golub,k=int(X.shape[1]*j[i]))
        filter_svm = make_pipeline(filter_selector,classifier)
        grid_search = GridSearchCV(filter_svm, param_grid=param_grid)
        result.append (np.mean(cross_validation.cross_val_score(grid_search, X, y, cv=cv)))
        
    return result

def part3(X,y):
    for i in range(10):
        j = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        cv = cross_validation.StratifiedKFold(y, 5, shuffle=True, random_state=0)
        filter_selector = SelectKBest(golub,k=int(X.shape[1]*j[i]))
        classifier = LinearSVC(penalty='l2',dual=False)
        filter_svm = make_pipeline(filter_selector, classifier)
        result.append (np.mean(cross_validation.cross_val_score(filter_svm, X, y, cv=cv)))
        
    return result
    
if __name__=='__main__' :
    leu = golub(X_train,y_train)
    leu_2 = chooseC(X_train,y_train)
    leu_3 = part3(X_train,y_train)
    print leu_2,'leu2',leu_3,'leu3'
    
    arcene = golub(X2,y)
    arcene_2 = chooseC(X2,y)
    arcene_3 = part3(X2,y)
    print arcene_2,arcene_3