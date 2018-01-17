# -*- coding: UTF-8 -*-
import numpy as np


class Perception(object):
    def __init__(self, eta=0.01, n_iter=10):
        """

        :param eta: 学习率
        :param n_iter: 权重向量的训练次数
        w_:神经分叉权重向量
        errors_:用于记录神经元判断出错的次数
        """
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        输入训练数据， 培训神经元
        :param X: 输入样本向量 shape[n_samples, n_features]
                    X:[[1, 2, 3], [4, 5, 6]]
                    n_samples: 2
                    n_features: 3
        :param y: 样本分类 y:[1, -1]
        :return:
        """

        """
        初始化权重向量为0
        加一是因为w0, 步调函数阈值
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ =[]

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1, 2, 3], [3, 4, 5]]
            y:[1, -1]
            zip(X, y) = [([1, 2, 3], 1), ([3, 4, 5], -1)]
            """
            # print(zip(X, y))
            for xi, target in zip(X, y):
                """
                update = η * （y - y')
                """
                update = self.eta * (target - self.predict(xi))

                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update != 0.0)
                pass
            self.errors_.append(errors)
        pass

    def net_input(self, X):
        """
        z = w0*1 + w1*x1 + ... + wn*xn
        :param X:
        :return:
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        pass
    pass
