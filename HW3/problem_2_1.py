import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def load_data(file_name):
    col_names = ['Diagnosis']
    for i in range(1, 23):
        col_names.append(f'F{i}')
    data = pd.read_csv(file_name, names=col_names)
    y = data['Diagnosis'].values
    x = data[col_names[1:]].values
    y[y == 0] = -1
    return x, y


class Node:
    def __init__(self, x, y, split_attr):
        self.split_attr = split_attr
        self.depth = 0
        self.zero = None
        self.one = None
        if len(y) == 0:
            self.prediction = None
            self.partition = None
            return
        self.prediction = np.unique(y, return_counts=True)
        self.prediction = self.prediction[0][self.prediction[1].argmax()]

        if self.split_attr == -1:
            self.partition = [[[], []], [[], []]]
        else:
            self.partition = [[x[(x[:, split_attr] == 0)], y[(x[:, split_attr] == 0)]],
                              [x[(x[:, split_attr] == 1)], y[(x[:, split_attr] == 1)]]]

    def copy(self):
        copy = Node([], [], self.split_attr)
        if self.one:
            copy.one = self.one.copy()
        if self.zero:
            copy.zero = self.zero.copy()
        copy.prediction = self.prediction
        copy.partition = self.partition
        copy.depth = self.depth
        return copy

    def set_zero(self, zero, shape=1):
        self.zero = zero
        self.zero.depth = self.depth + 1
        if shape != 0:
            self.one = Node(self.partition[1][0], self.partition[1][1], -1)
            self.one.depth = self.depth + 1

    def set_one(self, one, shape=1):
        self.one = one
        self.one.depth = self.depth + 1
        if shape != 0:
            self.zero = Node(self.partition[0][0], self.partition[0][1], -1)
            self.zero.depth = self.depth + 1

    def empty(self):
        if self.partition is None:
            self.one = Node([], [], -1)
            self.zero = Node([], [], -1)
        else:
            self.one = Node(self.partition[1][0], self.partition[1][1], -1)
            self.zero = Node(self.partition[0][0], self.partition[0][1], -1)
        self.zero.depth = self.depth + 1
        self.one.depth = self.depth + 1

    def part(self):
        return self.partition

    def predict(self, value):
        if self.split_attr == -1:
            return self.prediction

        prediction = None
        if value[self.split_attr] == 0 and self.zero:
            prediction = self.zero.predict(value)
        elif value[self.split_attr] == 1 and self.one:
            prediction = self.one.predict(value)
        if prediction is None:
            prediction = self.prediction
        return prediction

    def __str__(self):
        return f'Split: {self.split_attr + 1}, Predict: {self.prediction}' + \
            (f'\n\t' + '\t' * self.depth + f'ZERO: {self.zero}\n\t' + '\t' * self.depth + f'ONE:  {self.one}'
             if self.zero and self.one else '')


def iterate_trees(x, y, weights):
    num = 0
    best_acc = -1
    best_tree = None
    best_args = (0, 0, 0, 0)
    for i in range(5):
        for a1 in range(x.shape[1]):
            root = Node(x, y, a1)
            root_part = root.part()
            for a2 in range(x.shape[1]):
                if i in [0, 1, 2]:
                    node2 = Node(root_part[0][0], root_part[0][1], a2)
                    root.set_zero(node2, i)
                else:
                    node2 = Node(root_part[1][0], root_part[1][1], a2)
                    root.set_one(node2)
                node2_part = node2.part()
                for a3 in range(x.shape[1]):
                    if i == 0:
                        node3 = Node(root_part[1][0], root_part[1][1], a3)
                        root.set_one(node2, i)
                        node2.empty()
                        node3.empty()
                    elif i in [1, 4]:
                        node3 = Node(node2_part[0][0], node2_part[0][1], a3)
                        node2.set_zero(node3)
                    else:
                        node3 = Node(node2_part[1][0], node2_part[1][1], a3)
                        node2.set_one(node3)
                    node3.empty()
                    acc = accuracy(x, y, root, weights)
                    num += 1
                    if acc > best_acc:
                        best_acc = acc
                        best_tree = root.copy()
                        best_args = (i, a1 + 1, a2 + 1, a3 + 1)
    return best_tree, best_args


def generate_tree(x, y, i, a1, a2, a3):
    root = Node(x, y, a1)
    root_part = root.part()
    if i in [0, 1, 2]:
        node2 = Node(root_part[0][0], root_part[0][1], a2)
        root.set_zero(node2, i)
    else:
        node2 = Node(root_part[1][0], root_part[1][1], a2)
        root.set_one(node2)
    node2_part = node2.part()
    if i == 0:
        node3 = Node(root_part[1][0], root_part[1][1], a3)
        root.set_one(node2, i)
        node2.empty()
        node3.empty()
    elif i in [1, 4]:
        node3 = Node(node2_part[0][0], node2_part[0][1], a3)
        node2.set_zero(node3)
    else:
        node3 = Node(node2_part[1][0], node2_part[1][1], a3)
        node2.set_one(node3)
    node3.empty()
    return root


def accuracy(x, y, root, weights):
    correct = 0
    for xi, yi, wi in zip(x, y, weights):
        correct += wi * (root.predict(xi) == yi)
    return correct


def update_weights(x, y, root, weights, round, error, alpha):

    for i, w in enumerate(weights[round]):
        weights[round + 1][i] = w * math.exp(-y[i] * root.predict(x[i]) * alpha) / (2 * math.sqrt((1 - error) * error))


def combined_accuracy(x, y, trees, alphas, rounds):
    correct = 0
    for xi, yi in zip(x, y):
        curr = 0
        for i in range(rounds):
            curr += trees[i].predict(xi) * alphas[i]
        correct += (curr * yi >= 0)
    return correct / len(y)


x_train, y_train = load_data('heart_train.data')
x_test, y_test = load_data('heart_test.data')
rounds = 10
trees = []
weights = np.ones((rounds + 1, x_train.shape[0])) * (1 / x_train.shape[0])
alphas = [0.0] * rounds
errors = [0.0] * rounds


# args = [(1, 7, 10, 15), (1, 11, 19, 21), (1, 6, 18, 12), (4, 12, 0, 7), (1, 3, 16, 21)]

# tree = generate_tree(x_train, y_train, 1, 0, 0, 0)
# print(tree)
# exit(0)

x = range(1, rounds+1)
training_acc = []
test_acc = []

for r in range(rounds):
    print(f'ROUND: {r + 1}')
    tree, args = iterate_trees(x_train, y_train, weights[r])
    # tree = generate_tree(x_train, y_train, args[r][0], args[r][1], args[r][2], args[r][3])
    trees.append(tree)
    errors[r] = 1 - accuracy(x_train, y_train, tree, weights[r])
    alphas[r] = 0.5 * math.log((1 - errors[r]) / errors[r])
    update_weights(x_train, y_train, tree, weights, r, errors[r], alphas[r])
    # print(f'tree:{tree}')
    print(f'args:{args}')
    print(f'epsilon:{errors[r]}, alpha:{alphas[r]}')
    training_acc.append(combined_accuracy(x_train, y_train, trees, alphas, r+1))
    test_acc.append(combined_accuracy(x_test, y_test, trees, alphas, r+1))
    print(f'Training Accuracy:{training_acc[r]:<2.1%}, Test Accuracy:{test_acc[r]:<2.1%}')

plt.plot(x, training_acc, label='Training Accuracy')
plt.plot(x, test_acc, label='Test Accuracy')
plt.legend
plt.xlabel('Round')
plt.xticks(x)
plt.ylabel('Accuracy')
plt.title('Accuracy Vs Boosting Round')
plt.show()


