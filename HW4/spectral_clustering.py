import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

def circs():
    x = np.zeros((2, 100))
    for y in range(50):
        i = y * math.pi / 25
        x[0, y] = math.cos(i)
        x[1, y] = math.sin(i)
        x[0, y + 50] = 2 * math.cos(i)
        x[1, y + 50] = 2 * math.sin(i)
    return x


def generate_similarity_matrix(x, sigma):
    c = -1 / (2 * sigma ** 2)
    A = np.exp(c * np.square(x[:, None] - x).sum(axis=-1))
    return A


def spectral_clustering(x, sigma, k):
    if os.path.exists('vec' + str(sigma) + '_' + str(k) + '.gz'):
        V = np.loadtxt('vec' + str(sigma) + '_' + str(k) + '.gz')
    else:
        A = generate_similarity_matrix(x, sigma)
        D = np.diag(A.sum(axis=1))
        L = D - A
        values, vectors = np.linalg.eigh(L)
        V = vectors[:, :k]
        np.savetxt('vec' + str(sigma) + '_' + str(k) + '.gz', V)
    S = k_means(V, k)
    return S


def k_means(x, k):
    clf = KMeans(n_clusters=k)
    S = clf.fit_predict(x)
    return S


def scatter_plot(x, S):
    num_clusters = np.unique(S).size
    C = [[0]] * num_clusters
    for i in range(num_clusters):
        C[i] = x[np.where(S == i)]
    for c in C:
        plt.scatter(c[:, 0], c[:, 1])


def img_plot(S):
    S = S.reshape((75, 100))
    plt.imshow(S, cmap='gray')


plt.subplots(2, 2, figsize=(8, 8))

x = circs().T
sigma = .1
k = 2

plt.subplot(2, 2, 1).set_title('Spectral Clustering Circs')
S = spectral_clustering(x, sigma, k)
scatter_plot(x, S)

plt.subplot(2, 2, 2).set_title('K means Circs')
S = k_means(x, k)
plt.title = 'K means'
scatter_plot(x, S)

image_data = np.array(mpimg.imread('bw.jpg'), dtype='float64')
x = image_data.flatten()
sigma = 10
k = 2

plt.subplot(2, 2, 3).set_title('Spectral Clustering BW')
S = spectral_clustering(x, sigma, k)
img_plot(S)

plt.subplot(2, 2, 4).set_title('K means BW')
S = k_means(x.reshape(-1, 1), k)
img_plot(S)
plt.show()
