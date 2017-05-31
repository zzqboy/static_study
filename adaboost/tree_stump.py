# -*- coding:utf-8 -*-
"""
author: acrafter
date: 17-5-26
单个结点的树桩,训练过程为CART树的过程
"""
from __future__ import division
from itertools import izip

from numpy import *


class Stump(object):
    def __init__(self):
        self.s = None
        self.c_final = []
        self.se = None  # 当前这颗树的总平均损失
        self.se_i = []  # 每个数据的平均损失

    def fit(self, x, y):
        m = shape(x)[0]
        error_lst = []
        c_lst = []
        for i in range(1, m):  # i等于切分点s
            error = 0
            c_1 = sum(y[:i]) / i
            c_2 = sum(y[i:]) / (m - i)
            for x_i in range(i):
                error += (y[x_i]-c_1)**2
            for x_i in range(i, m):
                error += (y[x_i]-c_2)**2
            error_lst.append(error)
            c_lst.append([c_1, c_2])

        min_error = min(error_lst)
        min_index = error_lst.index(min_error)

        self.s = x[min_index]
        self.c_final = c_lst[min_index]
        self.get_se(x, y)
        # print self.error_lst

        return self.s, self.c_final  # 返回最小切分点s和c1 c2

    def get_se(self, x, y):
        """计算当前树平方损失值"""
        se = 0
        se_i = []
        for x_i, y in izip(x, y):
            pred = self.predict(x_i)
            se += (pred - y)**2
            se_i.append(y-pred)
        self.se_i = se_i
        self.se = se

    def predict(self, x):
        if x <= self.s:
            return self.c_final[0]
        else:
            return self.c_final[1]


class CombineTree(object):
    """将多个单节点树结合"""
    def __init__(self):
        self.stumps = []
        self.s = []
        self.c = []
        self.se = None
        self.se_i = None  # 每个数据点的残差，用于下一次的单节点树训练

    def add(self, stump):
        self.stumps.append(stump)
        self.s.append(stump.s)
        self.s.sort()
        self.s = list(set(self.s))
        c = zeros((len(self.s)+1, 1))
        for stump in self.stumps:
            id = self.s.index(stump.s)
            c[:id+1] += stump.c_final[0]
            c[id+1:] += stump.c_final[1]
        self.c = c

    def predict(self, x):
        for i, s in enumerate(self.s):
            if x <= s:
                return self.c[i]
        else:
            return self.c[-1]

    def get_se(self, x, y):
        """计算联合树平方损失值"""
        se = 0
        se_i = []
        for x_i, y in izip(x, y):
            pred = self.predict(x_i)
            se += (pred - y) ** 2
            se_i.append(y-pred)
        self.se_i = se_i
        self.se = se
        return self.se, self.se_i


if __name__ == '__main__':
    # 例8.2
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]
    x, y = array(x), array(y)
    # tree = Stump()
    # print tree.fit(x, y)
    # print tree.se, tree.se_i

    ctree = CombineTree()
    s1 = Stump()
    s1.s, s1.c_final = 6.5, [6.24, 8.91]
    s2 = Stump()
    s2.s, s2.c_final = 3.5, [-0.52, 0.22]
    s3 = Stump()
    s3.s, s3.c_final = 6.5, [0.15, -0.22]
    ctree.add(s1)
    print ctree.get_se(x, y)
    ctree.add(s2)
    print ctree.get_se(x, y)
    ctree.add(s3)
    print ctree.get_se(x, y)
