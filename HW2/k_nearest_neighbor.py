import numpy as np


class Knn:
    def __init__(self, x, y, k=1):
        self.x = x
        self.y = y
        self.k = k
        self.norm_x = self.normalize(self.x)

    def normalize(self, data):
        column_sum = np.ndarray.sum(self.x, axis=0)/self.x.shape[0]
        norm_data = data / column_sum

        return norm_data

    def find_closest_k(self, point):
        distances = np.ones(self.x.shape[0])
        for i, xm in enumerate(self.norm_x):
            distances[i] = np.linalg.norm(xm - point)
        sorted_dist = np.msort(distances)
        closest_k = (distances <= sorted_dist[self.k-1])

        return self.y[closest_k]

    def accuracy(self, x_test, y_test):
        x_test_norm = self.normalize(x_test)
        count = 0
        for xm, ym in zip(x_test_norm, y_test):
            closest_k = self.find_closest_k(xm)
            counts = np.count_nonzero(closest_k+1)
            prediction = 1 if counts > closest_k.shape[0]/2 else -1
            count += 1 if prediction == ym else 0
        return count
