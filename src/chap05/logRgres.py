# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 10:46:06 2018

@author: Administrator
"""
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = [];
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        '''
            testSet文件中有三列，分别是X1，X2和类别信息
            训练集的数据初始化为(1.0, X1, X2)
        '''
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))    #这个exp是numpy里面的，可以矩阵运算

'''
    本章节省略了大量的数学推导过程
    从代价函数，代价函数的偏导，偏导的矢量化等过程被省略了
    一方面可能是这个操作方案基本算是共识，大家都会选取这个方面
    另一方面也可能是推导过程太数学化了
    需要了解这段代码的背景需要参考代价函数等各方面的资料
    目前就先记住这个结论，直接运行即可
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights
    

'''
    画出数据集和Logistic回归最佳拟合直线
'''
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

    
'''
    随机梯度上升算法
    原梯度上升算法计算量很大，虽然结果很理想，但是性价比一般
    除了计算次数很多之外，还有大量的矩阵计算
    可以尝试简化一下，如果简化的结果可以接受，就没有必要那么复杂了
    在本书的数据集基础上，实际效果比完全的梯度上升要差一些
'''
def stocGradAscent0(dataMatrix, classLabels):
    '''
        加这一行代码是解决这个异常：
            TypeError: 'numpy.float64' object cannot be interpreted as an integer
    '''
    dataMatrix=array(dataMatrix)
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
    改进的随机梯度上升算法
    alpha的取值会随着迭代次数不断减小
    加快回归系数的收敛
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''
        加这一行代码是解决这个异常：
            TypeError: 'numpy.float64' object cannot be interpreted as an integer
    '''
    dataMatrix=array(dataMatrix)
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        '''
            python 3.x当中range不会直接返回数组，而是返回一个range对象
            这是一个可iterate的对象，但是不能直接进行数组操作
        '''
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights






