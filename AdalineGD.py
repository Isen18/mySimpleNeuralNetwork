# -*- coding: UTF-8 -*-
import numpy as np


class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50):
        """

        :param eta: 学习率
        :param n_iter: 训练次数
        """
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """

        :param X: 训练数据，二维数组[n_samples, n_features]
        :param y: 正确分类
        :return:
        """
        """
        w_: 权重向量
        error_: 每次迭代时，错误判断次数
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
            pass
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)

    pass
