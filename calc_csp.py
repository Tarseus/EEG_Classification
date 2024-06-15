'''
This script is used to calculate the CSP features from the raw data.'''

import scipy.io 
from mne.decoding import CSP

data_dir = './data/'
file_name = 'alignedX_dim0.mat'

X = scipy.io.loadmat(data_dir + file_name)['X'].astype('double')
X_train = X[:800]
X_test = X[800:]
y = scipy.io.loadmat(data_dir + 'allY_dim0.mat')['y'].ravel()
print(X_train.shape)

csp = CSP(n_components=32, reg=None, log=True, norm_trace=False)

print(X_train.shape, y.shape)
csp.fit(X_train, y)

X_train_csp = csp.transform(X_train)
print(X_train_csp.shape)
X_test_csp = csp.transform(X_test)
print(X_test_csp.shape)
scipy.io.savemat(data_dir + 'X_train_csp.mat', {'X': X_train_csp},{'y': y})
scipy.io.savemat(data_dir + 'X_test_csp.mat', {'X': X_test_csp})