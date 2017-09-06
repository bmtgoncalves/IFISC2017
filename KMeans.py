import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_data(nclusters, npoints):
    centroids = np.random.randint(0, 3*nclusters, [nclusters, 2])

    clusters = []

    for i in range(nclusters):
        xy = np.random.normal(centroids[i], .3, [npoints, 2])

        clusters.extend(list(xy))

    return np.array(clusters)

nclusters = 5
npoints = 100

data = generate_data(nclusters, npoints)

kmeans = KMeans(n_clusters=nclusters)
kmeans.fit(data)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

for i in range(nclusters):
    plt.plot(data[labels==i].T[0], data[labels==i].T[1], '.')

plt.plot(centroids.T[0], centroids.T[1], 'b*')
plt.savefig('KMeans.png')
