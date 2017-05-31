# -*- coding:utf-8 -*-
"""
author: acrafter
date: 17-5-24
最大熵的GIS算法实现，参考https://vimsky.com/article/714.html
"""
import sys
import math
from collections import defaultdict
from itertools import izip


class MaxEnt:
    def __init__(self):
        self._samples = []  # 样本集, 元素是[y,x1,x2,...,xn]的元组
        self._Y = set([])  # 标签集合,相当于去重之后的y
        self._numXY = defaultdict(int)  # Key是(xi,yi)对，Value是count(xi,yi)
        self._N = 0  # 样本数量
        self._n = 0  # 特征对(xi,yi)总数量
        self._xyID = {}  # 对(x,y)对做的顺序编号(ID), Key是(xi,yi)对,Value是ID
        self._C = 0  # 样本最大的特征数量,用于求参数时的迭代，见IIS原理说明
        self._ep_ = []  # 样本分布的特征期望值
        self._ep = []  # 模型分布的特征期望值
        self._w = []  # 对应n个特征的权值
        self._last_w = []  # 上一轮迭代的权值
        self._EPS = 0.01  # 判断是否收敛的阈值

    def load_data(self, filename):
        for line in open(filename, "r"):
            sample = line.strip().split("\t")
            if len(sample) < 2:  # 至少：标签+一个特征
                continue
            y = sample[0]
            X = sample[1:]
            self._samples.append(sample)  # label + features
            self._Y.add(y)  # label
            for x in set(X):  # set给X去重
                self._numXY[(x, y)] += 1

    def _init_params(self):
        self._N = len(self._samples)
        self._n = len(self._numXY)
        self._C = max([len(sample) - 1 for sample in self._samples])
        self._w = [0.0] * self._n
        self._last_w = self._w[:]
        self._sample_ep()

    def _convergence(self):
        for w, lw in zip(self._w, self._last_w):
            if math.fabs(w - lw) >= self._EPS:
                return False
        return True

    def _sample_ep(self):
        self._ep_ = [0.0] * self._n
        # 计算方法参见公式(20)， 计算样本数据的期望
        for i, xy in enumerate(self._numXY):
            self._ep_[i] = self._numXY[xy] * 1.0 / self._N
            self._xyID[xy] = i

    def _zx(self, X):
        # calculate Z(X), 计算方法参见公式(15)
        ZX = 0.0
        for y in self._Y:
            sum = 0.0
            for x in X:
                if (x, y) in self._numXY:
                    sum += self._w[self._xyID[(x, y)]]
            ZX += math.exp(sum)
        return ZX

    def _pyx(self, X):
        # calculate p(y|x), 计算方法参见公式(22)
        ZX = self._zx(X)
        results = []
        for y in self._Y:
            sum = 0.0
            for x in X:
                if (x, y) in self._numXY:  # 这个判断相当于指示函数的作用
                    sum += self._w[self._xyID[(x, y)]]
            pyx = 1.0 / ZX * math.exp(sum)
            results.append((y, pyx))
        return results

    def _model_ep(self):
        self._ep = [0.0] * self._n
        # 参见公式(21)
        for sample in self._samples:
            X = sample[1:]
            pyx = self._pyx(X)
            for y, p in pyx:
                for x in X:
                    if (x, y) in self._numXY:
                        self._ep[self._xyID[(x, y)]] += p * 1.0 / self._N

    def train(self, max_iter=1000):
        self._init_params()
        for i in range(0, max_iter):
            print "Iter:%d..." % i
            self._last_w = self._w[:]  # 保存上一轮权值
            self._model_ep()
            # 更新每个特征的权值
            for i, w in enumerate(self._w):
                # 参考公式(19)
                self._w[i] += 1.0 / self._C * math.log(self._ep_[i] / self._ep[i])
            print self._w
            # 检查是否收敛
            if self._convergence():
                break

    def predict(self, input):
        X = input.strip().split("\t")
        prob = self._pyx(X)
        return prob


if __name__ == "__main__":
    maxent = MaxEnt()
    maxent.load_data('weather_data.txt')   # 字段的各个意思为： play outlook temperature humidity windy
    maxent.train()
    print maxent.predict("sunny\thot\thigh\tFALSE")
    print maxent.predict("overcast\thot\thigh\tFALSE")
    print maxent.predict("sunny\tcool\thigh\tTRUE")
