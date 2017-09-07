#!/usr/bin/env pythonw

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

X_train = np.load('input/X_train.npy')
y_train = np.load('input/y_train.npy')

order = list(range(X_train.shape[0]))
np.random.shuffle(order)

X_train = X_train[order]
y_train = y_train[order]

X_test = np.load('input/X_test.npy')
y_test = np.load('input/y_test.npy')

order = list(range(X_test.shape[0]))
np.random.shuffle(order)

X_test = X_test[order]
y_test = y_test[order]

X_validate = X_test[:1000]
X_test = X_test[1000:]

y_validate = y_test[:1000]
y_test = y_test[1000:]

input_layer_size = X_train.shape[1]

X_train /= 255.
X_test /= 255.
X_validate /= 255.

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_validate)

print(accuracy(y_validate, y_pred))