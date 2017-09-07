#!/usr/bin/env pythonw

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

# Use 1000 points from the training set for the validation set
X_validate = X_train[:1000]
X_train = X_train[1000:]

y_validate = y_train[:1000]
y_train = y_train[1000:]

input_layer_size = X_train.shape[1]

X_train /= 255.
X_test /= 255.

def accuracy(y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

results_distance = []
results_uniform = []

for k in range(1, 21):
    neigh = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance')
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    acc = accuracy(y_test, y_pred)
    results_distance.append([k, acc])

    print(k, acc)

for k in range(1, 21):
    neigh = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    acc = accuracy(y_test, y_pred)
    results_uniform.append([k, acc])
    print(k, acc)

results_distance = np.array(results_distance)
results_uniform = np.array(results_uniform)

plt.plot(results_uniform.T[0], results_uniform.T[1], 'r-')
plt.plot(results_distance.T[0], results_distance.T[1], 'b-')
plt.legend(['uniform', 'distance'])
plt.ylabel('accuracy')
plt.xlabel('number of neighbors')
plt.savefig('knn.png')

neigh = KNeighborsClassifier(n_neighbors=4, metric='euclidean', weights='distance')
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_validate)
acc = accuracy(y_validate, y_pred)

print(acc)