'''
This script trains the models and saves them to disk.'''

import json
import numpy as np
from joblib import dump
from joblib import load
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from keras.callbacks import Callback
from keras.layers import Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from keras import backend as K
import scipy.io

data_dir = './data/'

X = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')
y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
X_test = scipy.io.loadmat(data_dir + 'X_test_csp.mat')['X'].astype('double')
X_train = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[:600]
y_train = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[:600]
X_val = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[600:]
y_val = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[600:]

models = [
    ('SVM', svm.SVC(), {'kernel': ['poly'], 'C': [100], 'gamma': ['auto']}),
    ('RF', RandomForestClassifier(), {'n_estimators': [100], 'max_depth': [None], 'min_samples_split': [10]}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors': [10], 'weights': ['distance'], 'algorithm': ['auto']}),
    ('NB', GaussianNB(), {'var_smoothing': [1e-9]}),
]

best_scores = {}
kfold_results_acc = {}

for name, model, params in models:
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X, y)
    print(f"Best parameters for {name}: {clf.best_params_}")
    print(f"Best score for {name}: {clf.best_score_}")
    best_scores[name] = clf.best_score_
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    kfold_results_acc[name] = scores.tolist()
    dump(clf, f'{name}.joblib')

kfold = KFold(n_splits=5, shuffle=True)
accuracy_list = []
accuracy_list2 = []

for train_index, test_index in kfold.split(X, y):
    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y[train_index], y[test_index]
    y_train = np.reshape(y_train, (-1, 1))
    y_val = np.reshape(y_val, (-1, 1))

    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracy_list.append(accuracy)

    X_train = np.expand_dims(X_train, axis=2)
    model2 = Sequential()
    model2.add(Conv1D(64, 3, activation='relu', input_shape=(32, X_train.shape[2])))
    model2.add(Conv1D(64, 3, activation='relu'))
    model2.add(MaxPooling1D(3))
    model2.add(Conv1D(128, 3, activation='relu'))
    model2.add(Conv1D(128, 3, activation='relu'))
    model2.add(GlobalAveragePooling1D())
    model2.add(Dropout(0.5))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model2.fit(X_train, y_train, batch_size=64, epochs=150)
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    accuracy_list2.append(accuracy)

kfold_results_acc['FC'] = accuracy_list
kfold_results_acc['CNN'] = accuracy_list2

with open('kfold_results_acc.json', 'w') as f:
    json.dump(kfold_results_acc, f)
