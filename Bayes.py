# -*- coding:utf-8 -*-
"""
author: zzqboy
date: 17-4-19
贝叶斯分类器,这里没有加平滑
"""
from __future__ import division

from itertools import izip
from collections import Counter


class Bayes(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.class_counter = Counter()
        self.pre_proba_counter = Counter()
        self.class_proba = {}
        self.pre_proba = {}

    def fit(self):
        for y, x in izip(self.y, self.x):
            self.class_counter[y] += 1
            for x_l in x:
                self.pre_proba_counter[(y, x_l)] += 1
        self.class_proba = {y: self.class_counter[y]/len(self.y) for y in self.class_counter}
        self.pre_proba.update({y: self.pre_proba_counter[y]/self.class_counter[y[0]] for y in
                                       self.pre_proba_counter})

    def predict(self, x):
        be_proba = {}
        for y in self.class_proba:
            y_proba = self.class_proba[y]
            for x_l in x:
                y_proba *= self.pre_proba[(y, x_l)]
            be_proba[y] = y_proba
        for y_class, be_proba in sorted(be_proba.iteritems(), key=lambda x: x[1], reverse=True):
            print y_class, be_proba


if __name__ == '__main__':
    data = [[1, 's'], [1, 'M'], [1, 'M'], [1, 's'], [1, 's'], [2, 's'], [2, 'm'], [2, 'm'],
            [2, 'l'], [2, 'l'], [3, 'l'], [3, 'm'], [3, 'm'], [3, 'l'], [3, 'l']]
    label = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    b = Bayes(data, label)
    b.fit()
    b.predict([2, 's'])
