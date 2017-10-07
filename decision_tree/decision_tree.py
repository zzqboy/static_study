# -*- coding:utf-8 -*-
"""
author: zzqboy
date: 17-4-26
决策树，id3
"""
from __future__ import division

import math
from itertools import izip
from collections import Counter, defaultdict

import numpy as np


class Node(object):
    def __init__(self, feature):
        self.feature = feature
        self.value = {}


class DecisionTree(object):
    def __init__(self):
        self._label_counter = Counter()  # 对应C_k
        self._feature_counter = defaultdict(Counter)  # 对应D_i
        self._feature_label_counter = defaultdict(Counter)  # 对应D_ik

        self._entropy = 0
        self.information_gain = {}

    def _fit(self, x, y):
        d = 0
        for x, y in izip(x, y):
            d += 1
            self._label_counter[y] += 1
            for feature, value in enumerate(x):
                self._feature_counter[feature][value] += 1
                self._feature_label_counter[(feature, value)][y] += 1

        for _, c_i in self._label_counter.iteritems():
            self._entropy += -1*(abs(c_i)/abs(d))*math.log(abs(c_i)/abs(d), 2)

        for feature, value_count in self._feature_counter.iteritems():
            _condition_value = 0
            value_count = value_count.iteritems()
            for value, count in value_count:
                for label, count2 in self._feature_label_counter[(feature, value)].iteritems():
                    _condition_value += (abs(count)/abs(d))*(abs(count2)/abs(count))*math.log(abs(count2)/abs(count), 2)
            _condition_value *= -1
            self.information_gain[feature] = self._entropy - _condition_value

        return True

    def build_tree(self, x, y):
        # 重新初始化
        self._label_counter = Counter()
        self._feature_counter = defaultdict(Counter)
        self._feature_label_counter = defaultdict(Counter)
        self._entropy = 0
        self.information_gain = {}

        self._fit(x, y)
        self.information_gain = sorted(self.information_gain.iteritems(), key=lambda x: x[1], reverse=True)
        print '各个特征的信息增益\n', decision_tree.information_gain

        feature = self.information_gain[0][0]
        print '选择第{0}个特征'.format(feature)
        node = Node(feature)

        for value, counter in self._feature_counter[feature].iteritems():
            if len(self._feature_label_counter[(feature, value)]) == 1:
                node.value[value] = self._feature_label_counter[(feature, value)].keys()[0]
                print '取值', value, '叶节点', node.value[value]
            else:
                temp = x[:, feature] == value
                x = x[temp]
                y = y[temp]
                node.value[value] = self.build_tree(x, y)

        return node


if __name__ == '__main__':
    # 书中的贷款申请数据表，数值化, 特征为 0,1,2,3
    # 青年，中年，老年 = 1,2,3
    # 有工作，无 = 1, 2
    # 有房子，无 = 1, 2
    # 信贷情况, 一般，好，非常好 = 1, 2, 3
    data = [[1, 2, 2, 1],
            [1, 2, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [2, 2, 2, 1],
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 2, 1, 3],
            [2, 2, 1, 3],
            [3, 2, 1, 3],
            [3, 2, 1, 2],
            [3, 1, 2, 2],
            [3, 1, 2, 3],
            [3, 2, 2, 1]]
    label = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
    data, label = np.array(data), np.array(label)

    decision_tree = DecisionTree()
    id3_tree = decision_tree.build_tree(data, label)

    """输出
    各个特征的信息增益
    [(2, 0.4199730940219748), (3, 0.36298956253708536), (1, 0.32365019815155616), (0, 0.08300749985576883)]
    选择第2个特征
    取值 1 叶节点 1
    各个特征的信息增益
    [(1, 0.9182958340544896), (3, 0.47385138961004514), (0, 0.2516291673878229), (2, 0.0)]
    选择第1个特征
    取值 1 叶节点 1
    取值 2 叶节点 0
    """