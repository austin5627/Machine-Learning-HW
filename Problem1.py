import numpy as np


def num_correct(w, b):
    count = 0
    for m in data:
        xm = m[0:4]
        y = m[4]
        count += 1 if y * (np.dot(w, xm) + b) > 0 else 0
    return count


def standard_gradient_descent(w, b, gamma):
    for t in range(0, 10000000):
        deltaw = 0
        deltab = 0
        correct = 0
        for m in data:
            xm = m[0:4]
            y = m[4]
            discriminator = (1 - y * (np.dot(w, xm) + b))
            gradw = 2 * y * xm * discriminator
            gradb = 2 * y * discriminator
            deltaw += gradw if discriminator >= 0 else 0
            deltab += gradb if discriminator >= 0 else 0
            correct += 1 if y * (np.dot(w, xm) + b) > 0 else 0
        w = w + deltaw*gamma/data.shape[0]
        b = b + deltab*gamma/data.shape[0]
        if correct == data.shape[0]:
            return [w, b]


def stochastic_gradient_descent(w, b, gamma):
    for t in range(0, 10000000):
        deltaw = 0
        deltab = 0
        k = data[t % data.shape[0]]
        xk = k[0:4]
        y = k[4]
        discriminator = (1 - y * (np.dot(w, xk) + b))
        gradw = 2 * y * xk * discriminator
        gradb = 2 * y * discriminator
        deltaw += gradw if discriminator >= 0 else 0
        deltab += gradb if discriminator >= 0 else 0

        w = w + deltaw*gamma/data.shape[0]
        b = b + deltab*gamma/data.shape[0]
        if t % data.shape[0] == 0:
            correct = num_correct(w, b)
            if correct == data.shape[0]:
                return [t, w, b]


data = np.loadtxt("perceptron.data", delimiter=",")
# standard_parameters = standard_gradient_descent(w=np.array([0, 0, 0, 0]), b=0, gamma=1)
stochastic_parameters = stochastic_gradient_descent(w=np.array([0, 0, 0, 0]), b=0, gamma=1)
print(stochastic_parameters)

