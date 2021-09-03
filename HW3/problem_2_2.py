import math
import pandas as pd
import numpy as np


def load_data(file_name):
    col_names = ['Diagnosis']
    for i in range(1, 23):
        col_names.append(f'F{i}')
    data = pd.read_csv(file_name, names=col_names)
    y = data['Diagnosis'].values
    x = data[col_names[1:]].values
    y[y == 0] = -1
    return x, y


class Stump:
    def __init__(self, x, y, split_attr, predictions):
        self.split_attr = split_attr
        self.predictions = predictions

    def predict(self, value):
        return self.predictions[value[self.split_attr]]

    def __str__(self):
        return f'Split:{self.split_attr}, Predictions:{self.predictions}'


def generate_stumps(x, y):
    stumps = []
    for a in range(x.shape[1]):
        for z in [-1, 1]:
            for o in [-1, 1]:
                stumps.append(Stump(x, y, a, [z, o]))
    return stumps


def loss(x, y, stumps, alphas):
    loss_total = 0
    for xi, yi in zip(x, y):
        exp_total = 0
        for alpha, stump in zip(alphas, stumps):
            exp_total += alpha * stump.predict(xi)
        loss_total += math.exp(-yi * exp_total)
    return loss_total


def update_alpha_n(x, y, alphas, n, stumps):
    numerator = 0
    denominator = 0
    for xi, yi in zip(x, y):
        exp_total = 0
        for i, (alpha, stump) in enumerate(zip(alphas, stumps)):
            if i != n:
                exp_total += alpha * stump.predict(xi)
        if stumps[n].predict(xi) == yi:
            numerator += math.exp(-yi * exp_total)
        else:
            denominator += math.exp(-yi * exp_total)
    alpha_n = .5 * math.log(numerator/denominator)
    return alpha_n


def coordinate_descent(x, y, stumps):
    alphas = np.array([0.0] * len(stumps))
    for n, _ in enumerate(alphas):
        for j in range(100000):
            alpha_n = update_alpha_n(x, y, alphas, n, stumps)
            if alpha_n == alphas[n]:
                break
            alphas[n] = alpha_n
    return alphas


def update_weights(x, y, stump, weights, round, error, alpha):
    for i, w in enumerate(weights[round]):
        weights[round + 1][i] = w * math.exp(-y[i] * stump.predict(x[i]) * alpha) / (2 * math.sqrt((1 - error) * error))


def get_best_stump(x, y, stumps, weights=None):
    if weights is None:
        weights = [1/x.shape[0]]*x.shape[0]
    best = (0, stumps[0])
    for s in stumps:
        acc = weighted_accuracy(x, y, weights, s)
        if acc > best[0]:
            best = (acc, s)
    return best[1]


def weighted_accuracy(x, y, weights, stump):
    correct = 0
    for xi, yi, wi in zip(x, y, weights):
        correct += wi * (stump.predict(xi) == yi)
    return correct


def ada_boost(x, y, stumps, rounds=20):
    weights = np.ones((rounds + 1, x_train.shape[0])) * (1 / x_train.shape[0])
    ada_stumps = []
    ada_errors = []
    ada_alphas = []
    for r in range(rounds):
        ada_stumps.append(get_best_stump(x, y, stumps, weights[r]))
        ada_errors.append(1 - weighted_accuracy(x, y, weights[r], ada_stumps[r]))
        ada_alphas.append(0.5 * math.log((1 - ada_errors[r]) / ada_errors[r]))
        update_weights(x, y, ada_stumps[r], weights, r, ada_errors[r], ada_alphas[r])
    return ada_stumps, np.array(ada_alphas)


def accuracy(x, y, stumps, alphas):
    correct = 0
    for xi, yi in zip(x, y):
        curr = 0
        for s, a in zip(stumps, alphas):
            curr += s.predict(xi) * a
        correct += (curr * yi >= 0)
    return correct/len(y)


def bagging(x, y, stumps, bags=20):
    bag_stumps = []
    for b in range(bags):
        shuffle = np.arange(x.shape[0])
        shuffle = np.random.choice(shuffle, x.shape[0])
        bag_x = x[shuffle]
        bag_y = y[shuffle]
        bag_stumps.append(get_best_stump(bag_x, bag_y, stumps))
    return bag_stumps


def bag_accuracy(x, y, bag_stumps):
    correct = 0
    for xi, yi in zip(x, y):
        curr = 0
        for s in bag_stumps:
            curr += s.predict(xi) * (1/len(bag_stumps))
        correct += (curr * yi >= 0)
    return correct/len(y)


x_train, y_train = load_data('heart_train.data')
x_test, y_test = load_data('heart_test.data')
stumps = generate_stumps(x_train, y_train)


print(f'{"Coordinate Descent":=^40}')
coord_alphas = coordinate_descent(x_train, y_train, stumps)
print(coord_alphas)
print(f'Training Accuracy: {accuracy(x_train, y_train, stumps, coord_alphas):<3.1%}')
print(f'Test Accuracy: {accuracy(x_test, y_test, stumps, coord_alphas):<3.1%}')
print(f'Training Loss: {loss(x_train, y_train, stumps, coord_alphas):<3.2f}')

print()
print(f'{"Ada Boost":=^40}')
ada_stumps, ada_alphas = ada_boost(x_train, y_train, stumps)
print(ada_alphas)
print(f'Training Accuracy: {accuracy(x_train, y_train, ada_stumps, ada_alphas):<3.1%}')
print(f'Test Accuracy: {accuracy(x_test, y_test, ada_stumps, ada_alphas):<3.1%}')

print()
print(f'{"Bagging":=^40}')
bag_stumps = bagging(x_train, y_train, stumps)
print(f'Training Accuracy: {bag_accuracy(x_train, y_train, bag_stumps):<3.1%}')
print(f'Test Accuracy: {bag_accuracy(x_test, y_test, bag_stumps):<3.1%}')
