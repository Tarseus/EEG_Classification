'''
This script is used to find the best hyperparameters for each model using GridSearchCV.'''

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
import os


data_dir = './data/'

X = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')
y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
X_test = scipy.io.loadmat(data_dir + 'X_test_csp.mat')['X'].astype('double')
X_train = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[:600]
y_train = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[:600]
X_val = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[600:]
y_val = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[600:]
models = [
    ('SVM', svm.SVC(), {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}),
    ('RF', RandomForestClassifier(), {'n_estimators': [50, 100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}),
    # ('LR_l1', LogisticRegression(penalty='l1', solver='liblinear'), {'C': [0.01, 0.1, 1, 10, 100]}),
    # ('LR_l2', LogisticRegression(penalty='l2'), {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}),
    # ('LR_elasticnet', LogisticRegression(penalty='elasticnet', solver='saga'), {'C': [0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.1, 0.5, 0.9]}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [3,5,7,10], 'weights': ['uniform','distance'], 'algorithm': ['auto']}),
    ('NB', GaussianNB(), {'var_smoothing': [1e-9]}),
]
best_scores = {}
kfold_results_acc = {}
kfold_results_f1 = {}
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)
def clean_filename(filename):
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    for c in invalid_chars:
        filename = filename.replace(c, '')
    return filename

for name, model, params in models:
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X,y)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    kfold_results_acc[name] = scores.tolist()

    # Save results for each parameter combination
    for i in range(len(clf.cv_results_['params'])):
        param_combination = clf.cv_results_['params'][i]
        mean_test_score = clf.cv_results_['mean_test_score'][i]
        std_test_score = clf.cv_results_['std_test_score'][i]

        result = {
            'params': param_combination,
            'mean_test_score': mean_test_score,
            'std_test_score': std_test_score
        }

        filename = save_dir + name + str(param_combination) + '_acc.json'
        filename = clean_filename(filename)

        with open(filename, 'w') as f:
            json.dump(result, f)