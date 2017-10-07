# -*- coding:utf-8 -*-
"""
author: zzqboy
date: 17-5-26
adaboost用决策树做基函数等于提升树
"""
from tree_stump import *

from numpy import *


class AdaBoost(object):
    def __init__(self):
        self.final_tree = None  # 最后组合成的决策树

    def fit(self, x, y, m_estimate=6):
        ctree = CombineTree()
        y_temp = y
        for i in range(m_estimate):
            tree = Stump()  # 学习单节点树
            tree.fit(x, y)
            ctree.add(tree)
            se, se_i = ctree.get_se(x, y_temp)

            x, y = x, se_i

            # print '迭代--------------', i
            # print 's', ctree.s
            # print 'c', ctree.c
            # print '平方误差', se
            # print '偏差', se_i
        self.final_tree = ctree

if __name__ == '__main__':
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]
    x, y = array(x), array(y)

    ada = AdaBoost()
    ada.fit(x, y)
    print ada.final_tree.s
    print ada.final_tree.c
    print ada.final_tree.se  # 最后的平方损失值
