# encoding:utf-8
"""
@author: zzqboy
@time: 2017/1/2 21:25
"""
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import animation


def loadDataSet(fileName):
    # 加载数据
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def clipAlpha(aj, H, L):
    # 保持alpha 在 [0, C]之间
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup):
    # 核函数
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


def calcWs(alphas, dataArr, classLabels):
    # 计算 W
    X = mat(dataArr);
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def calcEk(oS, k):
    # 计算误差 E
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    # 选择第二个alpha
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS, choose_temp):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
        (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H: print "L==H"; return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]  # changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        choose_temp.append(j)  # 记录选择的第二个点
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])  # update i by the same amount as j
        updateEk(oS, i)  # added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):  # full Platt SMO
    choose_all_x = []  # 用来记录每次选择的样本
    all_E = []  # 记录变化的误差
    all_alpha, all_b = [], []  # 记录变化的alpha和b
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                choose_temp = [i]
                alphaPairsChanged += innerL(i, oS, choose_temp)
                choose_all_x.append(choose_temp)
                all_alpha.append(oS.alphas)
                all_b.append(oS.b)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                choose_temp = [i]
                alphaPairsChanged += innerL(i, oS, choose_temp)
                choose_all_x.append(choose_temp)
                all_alpha.append(oS.alphas)
                all_b.append(oS.b)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged)
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif (alphaPairsChanged == 0):
            entireSet = True
        print "iteration number: %d" % iter
    return oS.b, oS.alphas, choose_all_x, all_alpha, all_b

def compute_all_e(dataMatrix, labelMat, alphas, b, m):
    # 计算总误差方法，实际要更新的只是 alpha , b
    e = 0
    for i in range(m):
        fXi = float(multiply(alphas, labelMat).T * dataMatrix * dataMatrix[i, :].T) + b  # 预测的类别
        Ei = fXi - float(labelMat[i])
        e += Ei
    return e

def draw_svm_learning():
    ### 开始拟合
    dataArr, labelArr = loadDataSet('testSet.txt')
    dataMatrix,  labelMat = mat(dataArr), mat(labelArr).transpose()
    b, alphas, choose_all_x, all_alpha, all_b = smoP(dataArr, labelArr, 0.6, 0.001, 40)

    ## 找到使得直线方程改变的两个点 alpha1 alpha2
    change_index = [index for index, i in enumerate(choose_all_x) if len(i) == 2]
    change_x, change_alpha, change_b = [], [], []
    for i in change_index:
        change_x.append(choose_all_x[i])
        change_alpha.append(all_alpha[i])
        change_b.append(all_b[i])
    print len(choose_all_x)
    print len(change_x)

    ### 下面开始画图
    fig = plt.figure(121)
    ax = plt.axes(xlim=(-2, 12), ylim=(-8, 6))
    line, = ax.plot([], [])

    # 标签为-1的点
    xcord0 = [dataArr[i][0] for i in range(len(labelArr)) if labelArr[i] == -1]
    ycord0 = [dataArr[i][1] for i in range(len(labelArr)) if labelArr[i] == -1]
    # 标签为1的点
    xcord1 = [dataArr[i][0] for i in range(len(labelArr)) if labelArr[i] == 1]
    ycord1 = [dataArr[i][1] for i in range(len(labelArr)) if labelArr[i] == 1]

    # 更新每次的超平面方程
    def animate(time):
        # time 代表迭代次数，也是帧数
        label = u'Learning itertime {0}'.format(time)
        ax.set_xlabel(label)
        a, b = change_alpha[time], change_b[time]
        w = calcWs(a, dataArr, labelArr)
        choose_x = change_x[time]
        # 画出每次优化的两个点和其他数据点
        for i in range(len(labelArr)):
            xPt = dataArr[i][0]
            yPt = dataArr[i][1]
            label = labelArr[i]
            if i in choose_x:
                continue
            if (label == -1):
                ax.scatter(xPt, yPt, marker='o', s=60, linewidths=0.01)
            else:
                ax.scatter(xcord1, ycord1, marker='o', s=90, c='red',linewidths=0.1)
        for i in choose_x:
            ax.scatter(dataArr[i][0], dataArr[i][1], marker='o', s=90, c='cyan',linewidths=0.1)
        # 画出直线
        w0 = w[0][0]
        w1 = w[1][0]
        b = float(b)
        x = arange(-2.0, 12.0, 0.1)
        y = (-w0 * x - b) / w1
        line.set_data(x, y)
        plt.title(u'SVM (zzqboy.com)')
        return line, ax

    anim = animation.FuncAnimation(fig, animate, frames=len(change_x), interval=1)
    # plt.show()
    anim.save('svm.gif', fps=2, writer='imagemagick')

def draw_e():
    ### 开始拟合
    dataArr, labelArr = loadDataSet('testSet.txt')
    m = len(dataArr)
    dataMatrix,  labelMat = mat(dataArr), mat(labelArr).transpose()
    b, alphas, choose_all_x, all_alpha, all_b = smoP(dataArr, labelArr, 0.6, 0.001, 40)

    ### 下面开始画图
    fig = plt.figure()
    iter_time = len(choose_all_x)
    ax = plt.axes()
    ax.set_ylabel(u'error value')
    ax.set_xlabel(u'iter time')
    ax.set_title(u'svm (zzqboy.com)')
    y = []
    for i in range(iter_time):
        e = compute_all_e(dataMatrix, labelMat, all_alpha[i], all_b[i], m)
        y.append(e)
    ax.plot(range(iter_time), y)
    plt.show()

if __name__ == '__main__':
    # draw_svm_learning()
    draw_e()