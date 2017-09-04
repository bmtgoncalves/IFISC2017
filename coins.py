#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

def flip_coins(flips = 1000000, bins=100):
    # Uninformative prior
    prior = np.ones(bins, dtype='float')/bins
    likelihood_heads = np.arange(bins)/float(bins)
    likelihood_tails = 1-likelihood_heads
    flips = np.random.choice(a=[True, False], size=flips, p=[0.75, 0.25])

    for coin in flips:
        if coin:  # Heads
            posterior = prior * likelihood_heads
        else:  # Tails
            posterior = prior * likelihood_tails

        # Normalize
        posterior /= np.sum(posterior)

        # The posterior is now the new prior
        prior = posterior

    return posterior

plt.plot(np.arange(100)/float(100), flip_coins(10))
plt.plot(np.arange(100)/float(100), flip_coins(100))
plt.plot(np.arange(100)/float(100), flip_coins(1000))
plt.plot(np.arange(100)/float(100), flip_coins(10000))
plt.plot(np.arange(100)/float(100), flip_coins(100000))
plt.legend([10, 100, 1000, 10000, 100000])
plt.show()