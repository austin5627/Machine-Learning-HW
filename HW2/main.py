import svms
import k_nearest_neighbor
import numpy as np
import pandas as pd


def load_data(file_name):
    col_names = ['y']
    for i in range(1, 23):
        col_names.append(f'x{i}')
    data = pd.read_csv(file_name, sep=",", names=col_names)
    y = data['y'].values
    x = data[col_names[1:]].values
    for i, y_i in enumerate(y):
        if y_i == 0:
            y[i] = -1
    return x, y


def run_svm_slack(x, y, x_val, y_val, x_test, y_test):
    cs = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    best = 0
    best = (0, None, None, 0)
    for c in cs:
        w, b = svms.svm_slack(x, y, c)
        curr_accuracy_train = svms.accuracy_slack(w, b, x, y) / x.shape[0]
        curr_accuracy = svms.accuracy_slack(w, b, x_val, y_val) / x_val.shape[0]
        print(f'c:{c:<22.0f} Training Acc. {curr_accuracy_train: <4.0%} Validation Acc. {curr_accuracy: <4.0%}')
        if best[0] < curr_accuracy:
                best = (curr_accuracy, w, b, c)
    print(f'Best c: {best[3]}')
    best_acc = svms.accuracy_slack(best[1], best[2], x_test, y_test) / x_test.shape[0]
    print(f'Accuracy using test data {best_acc:<4.0%}')


def run_svm_dual(x, y, x_val, y_val, x_test, y_test):
    cs = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    sigma_2 = [1e-1, 1e0, 1e1, 1e2, 1e3]
    best = (0, None, None, 0, 0)
    for c in cs:
        for s in sigma_2:
            lambdas, b = svms.svm_dual(x, y, s, c)
            curr_accuracy_train = svms.accuracy_gauss(lambdas, b, x, y, x, y, s) / x.shape[0]
            curr_accuracy = svms.accuracy_gauss(lambdas, b, x, y, x_val, y_val, s) / x_val.shape[0]
            if best[0] < curr_accuracy:
                best = (curr_accuracy, lambdas, b, s, c)
            print(f'c:{c:<9.0f} sigma:{s:<6.1f} Training Acc. {curr_accuracy_train: <4.0%} Validation Acc. {curr_accuracy: <4.0%}')
    print(f'Best c: {best[4]}, Best sigma:{best[3]}')
    best_acc = svms.accuracy_gauss(best[1], best[2], x, y, x_test, y_test, best[3]) / x_test.shape[0]
    print(f'Accuracy using test data {best_acc:<4.0%}')


def run_k_nearest_neighbor(x, y, x_val, y_val, x_test, y_test):
    ks = [1, 5, 11, 15, 21]
    best = (0, 0, None)
    for k in ks:
        knn = k_nearest_neighbor.Knn(x, y, k)
        curr_accuracy_train = knn.accuracy(x, y) / x.shape[0]
        curr_accuracy = knn.accuracy(x_val, y_val) / x_val.shape[0]
        print(f'k: {k:<22.0f}Training Acc. {curr_accuracy_train:<4.0%} Validation Acc. {curr_accuracy:<4.0%}')
        if best[0] < curr_accuracy:
            best = (curr_accuracy, k, knn)
    print(f'Best k: {best[1]}')
    best_acc = best[2].accuracy(x_test, y_test) / x_test.shape[0]
    print(f'Accuracy using test data {best_acc:<4.0%}')


x, y = load_data('park_train.data')
x_validation, y_validation = load_data('park_validation.data')
x_test, y_test = load_data('park_test.data')

print(f'{"Question 1.1":-^63}')
run_svm_slack(x, y, x_validation, y_validation, x_test, y_test)
print(f'{"Question 1.2":-^63}')
run_svm_dual(x, y, x_validation, y_validation, x_test, y_test)
print(f'{"Question 1.3":-^63}')
run_k_nearest_neighbor(x, y, x_validation, y_validation, x_test, y_test)


