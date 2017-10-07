# -*- coding:utf-8 -*-
"""
author: zzqboy
date: 17-5-10
用随机梯度下降法学习逻辑回归, 参考《机器学习实战》，但是感觉里面写的不好
"""
from __future__ import division
import random

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from data.load_data import load_data


class LR(object):
    def __init__(self):
        self.w = 0
        self.alpha = 0
        self.learned_point = []

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.e ** (-x))

    def fit(self, x, y, iter_time):
        """
        StocGradAscent
        输入为numpy.mat结构
        """
        m, n = x.shape
        w = np.ones((1, n)).transpose()
        i = 0

        while i < iter_time:
            index = range(m)
            random.shuffle(index)  # 随机化

            for j in range(m):
                index_j = index[j]
                alpha = 4 / (1.0 + i + j) + 0.01     # 减小学习步长
                h = self.sigmoid(np.dot(x[index_j], w))
                error = y[index_j] - h
                w += alpha * error * x[index_j].reshape((2, 1))  # 更新参数
            i += 1

        self.w = w
        return w

    def test(self, x, y):
        y_pred = []
        for x_i in x:
            temp = self.sigmoid(np.dot(x_i, self.w))
            temp = 1 if temp >= 0.5 else 0
            y_pred.append(temp)

        print accuracy_score(y_test, y_pred)
        print classification_report(y, y_pred)

if __name__ == '__main__':
    x, y = load_data()
    x_train, y_train, x_test, y_test = x[:80], y[:80], x[80:], y[80:]

    lr = LR()
    lr.fit(x_train, y_train, 20)
    lr.test(x_test, y_test)

    """
    效果是0.6， sklearn可以做到100%.不得不服
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print accuracy_score(y_test, y_pred)
    """
