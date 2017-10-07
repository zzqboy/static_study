# encoding:utf-8
"""
@author: zzqboy
@time: 2017/7/16 15:06
隐马可夫模型
"""
import numpy as np


class HmmModel(object):
    def __init__(self, a, b, pi, o):
        self.a = a
        self.b = b
        self.pi = pi
        self.o = o
        self.alpha = None
        self.beta = None
        self.delta = None
        self.psi = None

    def forward(self):
        # 前向算法, 可以返回该观测序列的概率
        self.alpha = np.zeros((self.a.shape[0], self.o.shape[0]))

        temp = self.pi
        for index, o_i in enumerate(self.o):
            self.alpha[:, index] = temp * self.b[:, o_i]

            temp = []
            for i in range(self.a.shape[0]):
                temp.append(np.sum(self.alpha[:, index] * self.a[:, i]))
            temp = np.array(temp)

        return np.sum(temp)

    def back(self):
        # 后向算法，可以返回该观测序列的概率
        self.beta = np.zeros((self.a.shape[0], self.o.shape[0]))
        self.beta[:, -1:] = 1

        for t in range(self.o.shape[0]-2, -1, -1):
            for i in range(self.a.shape[0]):
                self.beta[i, t] = np.sum(self.beta[:, t+1] * self.a[i, :] * self.b[:, self.o[t+1]])

        return np.sum(self.pi * self.b[:, self.o[0]] * self.beta[:, 0])

    def viterbi(self):
        # 维特比算法
        self.delta = np.zeros((self.a.shape[0], self.o.shape[0]))
        self.psi = np.zeros((self.a.shape[0], self.o.shape[0]))
        self.delta[:, 0] = self.pi * self.b[:, self.o[0]]
        self.psi[:, 0] = 0

        final_point = None  # 最终点和概率
        hidden_path = np.zeros(self.o.shape[0])

        for t_i in range(1, self.o.shape[0]):
            for a_i in range(self.a.shape[0]):
                # 计算最大概率
                tran_pro = self.delta[:, t_i - 1] * self.a[:, a_i]
                max_tran = np.max(tran_pro)
                self.delta[a_i, t_i] = max_tran*self.b[a_i, self.o[t_i]]

                # 记录前一个状态j
                tran_index = np.where(tran_pro == max_tran)[0][0]
                self.psi[a_i, t_i] = tran_index
                final_point = tran_index

        # 回溯
        hidden_path[-1] = final_point
        for i in range(self.o.shape[0]-2, -1, -1):
            hidden_path[i] = self.psi[int(hidden_path[i+1]), i+1]
        return hidden_path


if __name__ == '__main__':
    """例10.2"""
    a = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    b = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    pi = [0.2, 0.4, 0.4]
    o = [0, 1, 0]
    a, b, pi, o = np.array(a), np.array(b), np.array(pi), np.array(o)

    h = HmmModel(a, b, pi, o)
    print h.forward()
    print h.alpha
    h.back()
    print h.beta

    """例10.3"""
    # 索引是从0开始的，所以此处的2表示状态3
    print '维特比算法'
    print h.viterbi()
    print h.delta
    print h.psi
