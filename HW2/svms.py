import numpy as np
import cvxopt


def accuracy_slack(w, b, x_check, y_check):
    count = 0
    for xm, ym in zip(x_check, y_check):
        prediction = np.dot(w.reshape([22, ]), xm) + b
        if prediction * ym >= 0:
            count += 1
    return count


def accuracy_gauss(lambdas, b, x_train, y_train, x_check, y_check, sigma):
    count = 0
    for m, xm in enumerate(x_check):
        prediction = sum(lambdas[i] * y_train[i] * gaussian_kernel(x_train[i], xm, sigma) for i, l in enumerate(lambdas)) + b
        count += 1 if prediction * y_check[m] > 0 else 0
    return count


def svm_slack(x, y, c=1):  # 22ws 1b 78xi
    size, length = x.shape
    p = np.diag(np.append(np.ones(length), np.zeros(size + 1)))  # 101 * 101
    q = np.append(np.zeros(length + 1), np.ones(size) * c)

    g_1 = np.hstack((np.ones([size, length + 1,]), np.identity(size) * -1))
    g_2 = np.hstack((np.zeros([size, length + 1]), np.identity(size) * -1))

    """
    y(wx + b) > 1 - xi
    ywx + by + xi > 1
    xi > 0
    
    -yx1 -yx2 ... -y 1 0 0 ...
    -yx1 -yx2 ... -y 0 1 0 ...
    -yx1 -yx2 ... -y 0 0 1 ...
    ...
    0 0 0 0 0 ...  0 1 0 0 ...
    0 0 0 0 0 ...  0 0 1 0 ...
    0 0 0 0 0 ...  0 0 0 1 ...
    ...
    """
    for i in range(size):
        for j in range(length):
            g_1[i, j] = -x[i, j] * y[i]
        g_1[i, length] = -y[i]

    g = np.vstack((g_1, g_2))

    h = np.append(np.ones(size) * -1, np.zeros(size))
    p = cvxopt.matrix(p)
    q = cvxopt.matrix(q)
    g = cvxopt.matrix(g)
    h = cvxopt.matrix(h)
    cvxopt.solvers.options['show_progress'] = False
    solution = np.array(cvxopt.solvers.qp(p, q, g, h)['x'])
    return solution[0:22], solution[22]


def gaussian_kernel(x1, x2, sigma):  # we are given sigma squared so don't square it here
    diff = x1 - x2
    dot = -np.dot(diff, diff) / (2 * sigma)
    kernel = np.exp(dot)

    return kernel


def svm_dual(x, y, sigma=1.0, c=1):
    size = x.shape[0]
    p = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            p[i, j] = y[i] * y[j] * gaussian_kernel(x[i], x[j], sigma)

    q = np.ones(size) * -1
    g = np.vstack((np.identity(size) * -1, np.identity(size)))
    h = np.append(np.zeros(size), np.ones(size) * c)
    a = y.reshape(1, -1) * 1.0
    b = np.zeros(1)
    p = cvxopt.matrix(p)
    q = cvxopt.matrix(q)
    g = cvxopt.matrix(g)
    h = cvxopt.matrix(h)
    a = cvxopt.matrix(a)
    b = cvxopt.matrix(b)
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(p, q, g, h, a, b)
    lambdas = np.array(solution['x'])

    total = np.ndarray.sum(lambdas)
    mean = total/lambdas.size

    index = (lambdas > mean).flatten()

    sv = x[index]
    sv_y = y[index]
    b = 0
    for s_x, s_y in zip(sv, sv_y):
        b += s_y - np.ndarray.sum(np.array([lambdas[i] * y[i] * gaussian_kernel(x[i], s_x, sigma) for i, l in enumerate(lambdas)]))
    b /= len(sv)

    return lambdas, b
