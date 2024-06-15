'''
this file is used to test if the data is loaded correctly and the model is trained correctly.'''

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.model_selection import cross_val_score
import scipy.io
import numpy as np
import json

data_dir = './data/'
clf = svm.SVC()
X = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')
y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
X_test = scipy.io.loadmat(data_dir + 'X_test_csp.mat')['X'].astype('double')
X_train = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[:600]
y_train = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[:600]
X_val = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[400:600]
# y_val = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[600:]
y_val = scipy.io.loadmat(data_dir + 's3.mat')['y'].ravel()
print(X_val.shape, y_val.shape)

clf.fit(X,y)
y_pred = clf.predict(X_val)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))