#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy as np
from scipy import optimize

data = np.loadtxt(sys.argv[1])

x = data.T[0]
y = data.T[1]

meanx = np.mean(x)
meany = np.mean(y)

w = np.sum((x-meanx)*(y-meany))/np.sum((x-meanx)**2)
b = meany-w*meanx

print(w, b)

# We can also optimize the Likelihood expression directly
def likelihood(w):
    global x, y
    sigma = 1.0
    w, b = w

    return np.sum((y-w*x-b)**2)/(2*sigma)

w, b = optimize.fmin_bfgs(likelihood, [1.0, 1.0])

print(w, b)
