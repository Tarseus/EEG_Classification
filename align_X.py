import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from scipy.linalg import fractional_matrix_power

data_dir = './data/'
file_names = ['s1.mat','s2.mat','s3.mat','s4.mat','s5.mat','s6.mat','s7.mat']
train_files = file_names[:4]
test_files = file_names[4:]
X = scipy.io.loadmat(data_dir + 's1.mat')['X']
y = scipy.io.loadmat(data_dir + 's1.mat')['y']
rawX = np.empty((0, X.shape[1], X.shape[2]))
allY = np.empty((0, y.shape[1]))
alignedX = np.empty((0, X.shape[1], X.shape[2]))
nTrials = 0
# print(y is None)
empty = True
for s in range(len(file_names)):
    data = scipy.io.loadmat(data_dir + file_names[s])
    X = data['X']
    
    if rawX is None:
        rawX = X
    else:
        rawX = np.concatenate((rawX, X), axis=0)
    if 'y' in data:
        y = data['y']
        # allY = np.dstack((allY, y))
        if empty:
            allY = y
            empty = False
        else:
            allY = np.concatenate((allY, y), axis=1)
            # allY = np.concatenate((allY, y.ravel()))
    # nTrials = len(allY)
    nTrials = 200

    refEuclidean = np.mean([np.cov(X[i,:,:]) for i in range(X.shape[0])], axis=0)
    # sqrtRefEuclidean = np.linalg.inv(np.linalg.cholesky(refEuclidean))
    sqrtRefEuclidean = fractional_matrix_power(refEuclidean, -0.5)

    # XR = np.empty((nTrials, X.shape[1], X.shape[2]))
    XE = np.empty((nTrials, X.shape[1], X.shape[2]))
    for j in range(nTrials):
        XE[j,:,:] = np.dot(sqrtRefEuclidean, X[j,:,:])

    # alignedX = np.dstack((alignedX, XE))
    if alignedX is None:
        alignedX = XE
    else:
        alignedX = np.concatenate((alignedX, XE), axis=0)

print(alignedX.shape)
scipy.io.savemat(data_dir + 'alignedX_dim0.mat', {'X': alignedX})
print(allY.shape)
allY = allY.transpose()
print(allY.shape)
scipy.io.savemat(data_dir + 'allY_dim0.mat', {'y': allY})