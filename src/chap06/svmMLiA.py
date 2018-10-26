# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:31:50 2018
直接从完整版的Platt SMO算法开始
简化版的略过
SVM的数学推导过程比较复杂，牵涉概念很多，甚至到了泛函论方面的相关知识
本章的学习就简单一点，直接了解结果
从Platt SMO算法开始，直接演练代码


@author: Administrator
"""

import numpy as np

'''
    辅助函数
'''
def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
    随机选择一个alpha
    i是第一个alpha的下标，m是所有alpha的数目
'''
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


################################ END ####################################

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
    
def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelsMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]







