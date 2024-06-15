'''
This script is used to validate the CSP parameter n_components.
'''

import scipy.io 
from mne.decoding import CSP
from sklearn import svm
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
from keras.models import Sequential
from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import load_model
from joblib import dump
from joblib import load
from sklearn.metrics import f1_score
from keras.callbacks import Callback

data_dir = './data/'
file_name = 'alignedX_dim0.mat'

X = scipy.io.loadmat(data_dir + file_name)['X'].astype('double')[:800]

y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
save_dir = './CSP_param_results/'
import os
os.makedirs(save_dir, exist_ok=True)

for n_components in [4,12,32,59]:
    model, params = (svm.SVC(), {'kernel': ['poly'], 'C': [100], 'gamma': ['auto']})
    X = scipy.io.loadmat(data_dir + file_name)['X'].astype('double')[:800]
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    # print(X.shape)
    csp.fit(X, y)
    X = csp.transform(X)
    # scipy.io.savemat(data_dir + 'X_train_csp_' + str(n_components) + '.mat', {'X': X_train_csp},{'y': y})
    # scipy.io.savemat(data_dir + 'X_test_csp_' + str(n_components) + '.mat', {'X': X_test_csp})
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    with open(save_dir + 'nFeatrues_' + str(n_components) + '_acc.json', 'w') as f:
        json.dump(scores.tolist(), f)
    