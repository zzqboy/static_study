# -*- coding:utf-8 -*-
"""
author: acrafter
date: 17-4-6
《统计学习方法》KNN算法
"""
import numpy as np


class Node(object):
    def __init__(self, value):
        self.right = None
        self.left = None
        self.parent = None
        self.value = value
        self.visited = False

    def set_left(self, left):
        left.parent = self
        self.left = left

    def set_right(self, right):
        right.parent = self
        self.right = right


def median(x):
    """x是某一维的array"""
    m = x.shape[0] / 2
    return x[m], m


def build_tree(x, k, j=0):
    """构建kdtree"""
    # 排序l维度并找到中位数
    l = j % k + 1
    temp = np.array(sorted(x, key=lambda x: x[l-1]))
    m_value, m = median(temp)
    print '构建树，以节点{}划分，深度{}'.format(m_value, j)

    kd_tree = Node(m_value)

    # 考虑3种情况，来切分子树
    if 0 < m < x.shape[0]-1:
        kd_tree.set_left(build_tree(temp[:m, :], k, j+1))
        kd_tree.set_right(build_tree(temp[m+1:, :], k, j+1))
    elif m == 0 and x.shape[0] > 1:
        kd_tree.set_right(build_tree(temp[m+1:, :], k, j+1))
    elif m == x.shape[0] - 1 and x.shape[0] > 1:
        kd_tree.set_left(build_tree(temp[:m, :], k, j+1))

    return kd_tree


def comp_distance(node, point):
    """l2距离"""
    v1, v2 = node.value, point
    return np.sqrt(sum((v1-v2)*(v1-v2)))


def find_near_parent(node, target, j, k):
    """前向找到最相近的父节点"""
    l = j % k
    current_value = node.value

    if target[l] < current_value[l]:
        if node.left:
            return find_near_parent(node.left, target, j+1, k)
    else:
        if node.right:
            return find_near_parent(node.right, target, j+1, k)

    print '找到最初相近点: ', node.value
    return node


def search(node, target, j, k, point_dist, s_node):
    """kd树查找"""
    if s_node.visited:
        return

    l = j % k
    max_dist = point_dist['Dist']

    # 已经找到包含目标的叶节点, 以此为当前最近点计算距离，这也是圆的距离
    c_dist = comp_distance(s_node, target)
    print '计算 {} 和 {} 的距离 {} \n'.format(str(s_node.value), str(target), c_dist)

    if c_dist < max_dist:
        max_dist = c_dist
        point_dist.update({'Point': node.value, 'Dist': max_dist})

    # 回溯到父节点
    p_point = s_node.parent
    s_node.visited = True
    while p_point and not p_point.visited:
        p_point.visited = True
        l_dist = abs(p_point.value[l]-node.value[l])

        # 判断另一子区域是否相交，找到另一子领域的相近点
        if l_dist < c_dist:

            p_dist = comp_distance(p_point, target)
            print '计算 {} 和 {} 的距离 {} \n'.format(str(p_point.value), str(target), p_dist)
            if p_dist < max_dist:
                max_dist = p_dist
                point_dist.update({'Point': p_point.value, 'Dist': max_dist})

            if p_point.left and not p_point.left.visited:
                return search(p_point.left, target, j, k, point_dist, s_node)
            if p_point.right and not p_point.right.visited:
                return search(p_point.right, target, j, k, point_dist, s_node)

        if p_point.parent and not p_point.parent.visited:
            p_point = p_point.parent
        else:
            break  # 已经回到根节点

    return point_dist


def knn():
    """利用kdtree找到topn个"""
    data = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
    target = np.array([9, 4])
    k = data.shape[1]

    kd_tree = build_tree(data, k)
    result = {'Point': None, 'Dist': float('inf')}
    s_node = find_near_parent(kd_tree, target, 0, k)
    search(kd_tree, target, 0, k, result, s_node)
    print result

if __name__ == '__main__':
    knn()
