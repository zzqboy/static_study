# -*- coding:utf-8 -*-
"""
author: acrafter
date: 17-5-12
加载数据
"""
import os
import codecs

import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))
testset_path = os.path.join(data_dir, 'testset')


def load_data(type='numpy'):
    x, y = [], []
    with codecs.open(testset_path, 'r', 'utf-8') as reader:
        for line in reader:
            line = line.strip()
            line = line.split('\t')

            temp = [float(line[0]), float(line[1])]

            x.append(temp)
            y.append(int(line[2]))

    if type == 'numpy':
        x, y = np.array(x), np.array(y)
    elif type == 'list':
        pass
    return x, y
