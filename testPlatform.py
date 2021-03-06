# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from Perception import Perception
from AdalineGD import AdalineGD


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    # print(x1_min, x1_max)
    # print(x2_min, x2_max)

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # print(np.arange(x1_min, x1_max, resolution).shape)
    # print(np.arange(x1_min, x1_max, resolution))
    # print(xx1.shape)
    # print(xx1)
    #
    # print(np.arange(x2_min, x2_max, resolution).shape)
    # print(np.arange(x2_min, x2_max, resolution))
    # print(xx2.shape)
    # print(xx2)

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # print(xx1.ravel())
    # print(xx2.ravel())
    # print(Z)

    Z = Z.reshape(xx1.shape)
    # print(Z)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # for idx, cl in enumerate(np.unique(y)):
    #     plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    plt.xlabel(u'花瓣长度')
    plt.ylabel(u'花径长度')
    plt.legend(loc='upper left')
    plt.show()
    pass


def test_perception(X, y):
    ppn = Perception(eta=0.1, n_iter=10)
    ppn.fit(X, y)
    plot_decision_regions(X, y, ppn, resolution=0.02)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('sum-sguard-error')
    plt.show()
    pass


def test_adaline_gd(X, y):
    ada = AdalineGD(eta=0.0001, n_iter=50)
    ada.fit(X, y)
    plt.title("Adaline-Grandient descent")
    plot_decision_regions(X, y, classifier=ada)

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('sum-sguard-error')
    plt.show()
    pass


def test():
    # filePath = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    # filePath = 'G:/PycharmProjects/nn_1/file/iris.csv'

    file_path = 'file/iris.csv'
    df = pd.read_csv(file_path, header=None)
    # print(df)
    # print(df.head(100))

    y = df.loc[0:99, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0, 2]].values
    # print(X)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # plt.show()

    test_perception(X, y)
    # test_adaline_gd(X, y)
    pass


test()
