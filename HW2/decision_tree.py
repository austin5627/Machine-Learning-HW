import pandas as pd
import numpy as np
import math


class Node:
    def __init__(self, x, y, depth=0, max_depth=-1):
        self.entropy = get_entropy(y)
        self.depth = depth
        self.prediction = np.unique(y, return_counts=True)
        self.prediction = self.prediction[0][self.prediction[1].argmax()]
        self.info_gain = 0

        if depth == max_depth or self.entropy == 0:
            self.children = None
            return
        else:
            self.split_attr, self.info_gain = decide_split(x, y)
            self.values = np.unique(x[:, self.split_attr])

            self.children = np.empty(self.values.shape)
            self.children = self.children.astype(Node)
            for i, v in enumerate(self.values):
                x_i = x[(x[:, self.split_attr] == v)]
                y_i = y[(x[:, self.split_attr] == v)]
                self.children[i] = Node(x_i, y_i, depth + 1, max_depth)

    def next_node(self, value):
        node = self.children[(self.values == value[self.split_attr])]
        if node.size == 0:
            return None
        return node[0]

    def predict(self, value):
        if self.children is None:
            return self.prediction
        else:
            node_next = self.next_node(value)
            if node_next is not None:
                return node_next.predict(value)
            else:
                return self.prediction

    def print(self, attr=0, value=''):
        if not value == '':
            print(f'{names[attr + 1] + "=" + value:19} ', end='')
        else:
            print(' ' * 20, end='')
        print(self)
        if self.children is not None:
            for i, v in zip(self.children, self.values):
                i.print(self.split_attr, v)

    def __str__(self):
        if self.children is not None:
            return f'splitting on {names[self.split_attr + 1]:<17} depth = {self.depth}, {self.info_gain}'
        else:
            return f'leaf node - prediction:{self.prediction:<7} depth = {self.depth}'


class DecisionTree:
    def __init__(self, x, y, max_depth=-1):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.root = Node(self.x, self.y, 0, self.max_depth)

    def predict(self, value):
        return self.root.predict(value)

    def visualize(self):
        self.root.print()


def get_entropy(y):
    h_y = 0
    for i in np.unique(y):
        p_y = np.count_nonzero(y == i) / y.shape[0]
        h_y -= p_y * math.log(p_y) if not p_y == 0 else 0
    return h_y


def get_info_gain(x, y, attribute):
    h_y = get_entropy(y)

    h_yx = 0
    for i in np.unique(x[:, attribute]):
        p_x = np.count_nonzero(x[:, attribute] == i) / x.shape[0]
        h_yx += p_x * get_entropy(y[(x[:, attribute] == i)])
    return h_y - h_yx


def decide_split(x, y):
    best_split = 0
    best_info = 0
    for a in range(x.shape[1]):
        info = get_info_gain(x, y, a)
        if info >= best_info:
            best_split = a
            best_info = info
    return best_split, best_info


def accuracy(tree, x, y):
    count = 0
    for xi, yi in zip(x, y):
        count += tree.predict(xi) == yi
    return count


names = [
    "poisonous?",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises?",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

data = pd.read_csv("mush_train.data", names=names)
x_train = data[names[1:]].values
y_train = data[names[0]].values
test_data = pd.read_csv("mush_test.data", names=names)
x_test = test_data[names[1:]].values
y_test = test_data[names[0]].values

d_tree = DecisionTree(x_train, y_train)
print(accuracy(d_tree, x_test, y_test) / x_test.shape[0])
d_tree.visualize()
