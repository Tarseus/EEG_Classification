'''
This script is used to predict the labels of the test data using the trained model.'''

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

X = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')
y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
X_test = scipy.io.loadmat(data_dir + 'X_test_csp.mat')['X'].astype('double')
X_train = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[:600]
y_train = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[:600]
X_val = scipy.io.loadmat(data_dir + 'X_train_csp.mat')['X'].astype('double')[600:]
y_val = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()[600:]

model = Sequential()
model.add(Dense(512, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = svm.SVC(C=100, gamma='auto', kernel='poly')
y = np.reshape(y, (-1, 1))
model.fit(X, y)

y_pred_s5 = model.predict(X_test[:200])
y_pred_s6 = model.predict(X_test[200:400])
y_pred_s7 = model.predict(X_test[400:600])
import numpy as np
y_pred = np.stack((y_pred_s5, y_pred_s6, y_pred_s7), axis=-1)
y_pred = np.squeeze(y_pred)
import pandas as pd
df = pd.DataFrame(y_pred,columns = ['s5','s6','s7'])
df.to_csv('y_pred.csv',index=False)
print('done!')