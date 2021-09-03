import math

import pandas as pd
import numpy as np
import cvxopt


def accuracy_slack(w, b, x_check, y_check):
    count = 0
    for xm, ym in zip(x_check, y_check):
        prediction = np.dot(w, xm) + b
        if prediction * ym >= 0:
            count += 1
    return count / len(y_check)


def error_slack(w, b, x_check, y_check):
    return 1 - accuracy_slack(w, b, x_check, y_check)


def svm_slack(x, y, c=1):
    size, length = x.shape
    p = np.diag(np.append(np.ones(length), np.zeros(size + 1)))
    q = np.append(np.zeros(length + 1), np.ones(size) * c)

    g_1 = np.hstack((np.ones([size, length + 1, ]), np.identity(size) * -1))
    g_2 = np.hstack((np.zeros([size, length + 1]), np.identity(size) * -1))

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
    w, b = solution[:length], solution[length]
    w = w.reshape(len(w), )
    return w, b


def get_eigens(x, k):
    mean = x.mean(axis=0)
    w = x - mean
    u, s, vh = np.linalg.svd(w)
    v = vh.T
    return v[:, :k], s[:k] ** 2


def run_pca_svm():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data('madelon.data')
    print(f'First 6 Eigen values of Covariance Matrix')
    vectors, values = get_eigens(x_train, 6)
    for i in values:
        print(f'{i:<1.2e}')

    k_list = [1, 2, 3, 4, 5, 6]
    c_list = [1, 10, 100, 1000]
    best = (1, None, None, 0, 0)
    print(f'SVM with PCA')
    mean = x_train.mean(axis=0)
    for k in k_list:
        eigen, _ = get_eigens(x_train, k)
        w = x_train - mean
        w_val = x_valid - mean
        x_pca_train = w @ eigen
        x_pca_valid = w_val @ eigen
        for c in c_list:
            svm_w, svm_b = svm_slack(x_pca_train, y_train, c)
            curr_error = error_slack(svm_w, svm_b, x_pca_valid, y_valid)
            if best[0] > curr_error:
                best = (curr_error, svm_w, svm_b, k, c)
            print(f'k:{k}, c:{c:<4} Error on validation set:{curr_error:<3.2%}')
    best_k, best_c = best[3], best[4]
    best_w, best_b = best[1], best[2]
    print(f'Best k:{best_k}, Best c:{best_c}')
    eigen_test, _ = get_eigens(x_train, best_k)
    x_pca_test = (x_test - mean) @ eigen_test
    test_error = error_slack(best_w, best_b, x_pca_test, y_test)
    print(f'Accuracy on test data {test_error:<3.2%}')


def run_pca_feature_selection():
    print('PCA for Feature Selection')
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data('madelon.data')
    k_list = [1, 10, 20, 40, 80, 160]
    c_list = [1, 10, 100, 1000]
    ex_num = 20
    best = (1, 0, 0, None)
    for k in k_list:
        s = int(k * math.log(k)) if k != 1 else 1
        vectors, values = get_eigens(x_train, k)
        pi = (vectors ** 2).sum(axis=1) / k
        print(f'k = {k}')
        for c in c_list:
            curr_error = np.zeros(ex_num)
            for ex in range(ex_num):
                samples = np.random.choice(np.arange(0, 500), p=pi, size=s, replace=True)
                samples = np.unique(samples)
                x_pca_train = x_train[:, samples]
                x_pca_valid = x_valid[:, samples]
                w, b = svm_slack(x_pca_train, y_train, c)
                curr_error[ex] = error_slack(w, b, x_pca_valid, y_valid)
            if best[0] > curr_error.mean():
                best = (curr_error.mean(), c)
            print(f'\tc:{c:<4} Avg Error on validation set:{curr_error.mean():<3.2%}')
        best_c = best[1]
        test_error = np.zeros(ex_num)
        for ex in range(ex_num):
            samples = np.random.choice(np.arange(0, 500), p=pi, size=s, replace=True)
            samples = np.unique(samples)
            x_pca_train = x_train[:, samples]
            x_pca_test = x_test[:, samples]
            w, b = svm_slack(x_pca_train, y_train, best_c)
            test_error[ex] = error_slack(w, b, x_pca_test, y_test)
        print(f'\tAvg Errir on test data with k={k}, c={best_c}: {test_error.mean():<3.2%}')


def load_data(file_name):
    data = pd.read_csv(file_name, header=None)
    x_data = data.values[:, : -1]
    y_data = data.values[:, -1]
    size, length = x_data.shape
    train_index = int(size * .6)
    valid_index = int(train_index + size * .3)

    x_train = x_data[:train_index]
    x_valid = x_data[train_index:valid_index]
    x_test = x_data[valid_index:]
    y_train = y_data[:train_index]
    y_valid = y_data[train_index:valid_index]
    y_test = y_data[valid_index:]
    return x_train, y_train, x_valid, y_valid, x_test, y_test


# run_pca_svm()
run_pca_feature_selection()
