# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 14:21:09 2018

@author: Administrator
"""

'''
    熵计算方法
'''

from math import log
import operator

def clcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #featVec是个向量，前面字段是特征，最后一个是类别,这里直接获取最后一个类别信息
        #此处用类别信息作为香农熵的计算输入
        currentLabel = featVec[-1] 
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1    
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'],[1, 0, 'no'],
               [0, 1, 'no'],[0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
    按照特征划分数据集
    dataSet：待划分的数据集，比如上面的dataSet
    axis：划分数据集的特征——其实是特征在特征向量中的下标位置
    value：需要返回的特征的值——其实是预期匹配到的特征值
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1 :])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
'''
    选择最好的数据集划分方式   
    返回值是最适合作为划分规则的特征在特征向量中的下标位置
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 
    baseEntropy = clcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * clcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
    一个工具方法
    由createTree方法内部调用
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), 
                                  key = operator.itemgetter(1), 
                                  reverse = True)
    return sortedClassCount[0][0]

'''
    递归构建决策树
    机器学习的学习过程就在这里体现
    可以隔一段时间，根据不同的材料重新构成决策树
    myTree：通过训练形成的决策树，其数据结构是多层嵌套的hash表（python里面叫字典）
        {label_1: 
            {feat_1: class_1, feat_2 : 
                {label_2: {feat_1: class_1, feat_2: class_2}}}}}}
    dataSet: （特征向量+标志）形成的列表
        比如：[[1, 1, 'yes'], [1, 1, 'yes'],[1, 0, 'no'],
               [0, 1, 'no'],[0, 1, 'no']]
    labels：特征向量的列名
'''
   
def createTree(dataSet, labels):
    '''
        取dataSet的每一个元素的最后一个标量组成一个列
        这里dataSet每个元素是形同[1, 1, 'yes']形式的一个列表
        最后一个标量就是'yes'这样的值，即分类（class）标志
    '''
    classList = [example[-1] for example in dataSet]
    '''
        统计classList第0个元素在classList当中的数量
        如果数量和classList的长度相等，则说明classList里面全部元素均相同
        即，类别完全相同，此时可以停止继续划分
    '''
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    '''
       如果仅剩下形如[['no'],['yes']...]这样的数据，表示已经把所有的特征都用尽了
       此时yes和no之类的类别还是都混杂在一起的话，可以选择占比高的进行返回
       以此作为最后的叶子节点
    '''
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最合适进行划分的属性下标
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 构造树的根节点,并且初始化数据结构为字典
    myTree = {bestFeatLabel: {}}
    # 删除已经成为根节点的label
    # 对比书上示例代码，这里先创建副本，免得修改了入参的数据结构
    newLabels = labels[:]
    del(newLabels[bestFeat])
    # 选取指定下标的特征值进行划分
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = newLabels[:]
        # 递归，形成嵌套的字典数据结构
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
              subLabels)
    return myTree

'''
    使用决策树进行分类操作
    inputTree：决策树
    featLabels：列名
    testVec：用于测试的特征向量
'''
def classify(inputTree, featLabels, testVec):
    #原书代码没有list，python3需要调用list方法才能把keys方法的结果转换为列表
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        # 特征向量的指定标量值和树的某个节点标签值匹配
        if testVec[featIndex] == key:
            # 如果此节点是一个字典，即是一棵子树，需要递归匹配
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                # 反之，直接返回节点的值即可
                classLabel = secondDict[key]
    return classLabel
    
 

    
    
    
    
    
    
        
        
        
        
        
        
        
        
        