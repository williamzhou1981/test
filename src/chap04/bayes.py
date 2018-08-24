# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 10:45:20 2018

@author: Administrator
"""
from numpy import *

'''
    创建了一些实验样本，模拟一个宠物论坛的留言
    标点符号已经去掉了，省点事
    文本是否有侮辱性采用人工标注的方法，用于以后的训练程序
'''
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

'''
    创建一个包含所有文档中出现的不重复词的列表
    用set最好，集合自动去重
    理论上讲，这个集合会很大，应该包括相当多的词，覆盖常用语
    这里的词汇集合来自上一个方法产生的数据集，不过个人认为这个不是必须的
    可以根据实际情况产生词汇集，比如从某些数据源（搜狗字库之类的）直接获取
    
    因为所有的算法处理过程都牵涉到计算，原始数据是字符串，无法参与到计算过程
    需要把字符串或者字符列表转换为可以参与计算的格式，比如数字向量
    此时这个vocabList就有用了
    参考下一个方法setOfWords2Vec
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''
    把文档转换为数字向量，和createVocabList方法紧密合作
    以createVocabList方法中生成的集合作为一个模板
    输入文档中的词如果在集合中，则对应位置置为1，反之为0
    生成一个数字向量，把文档从一个字符列表转化为可以参与计算的元素
    这里有些问题
        1、如果词汇表不全，那么可能导致丢失重要的信息
        2、如果词汇表太全，那么导致整个数字向量太长，甚至有可能内存放不下词汇表
        怎样折中解决呢？
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary" % word)
    return returnVec

'''
    下面的方法是利用已知的材料进行机器学习训练
    这个场景中贝叶斯公式如下：
        p(c_i | w) = p(w | c_i) p(c_i) / p(w)
            c_i是类别，侮辱性和非侮辱性留言两大类
                可以简单的通过各类文档数除以文档总数来获得p(c_i)
            w是setOfWords2Vec方法生成的文档向量
                直接计算比较复杂，可以利用贝叶斯假设
                    即每个向量的分量彼此独立，每个分量可以单独参与计算，如下
                    p(w | c_i) = [p(w_0 | c_i),....p(w_n | c_i)]
                    这样简化计算过程，当然最后还是以向量的形式进行展示
            p(w)直接用文档本身数量除以文档总数即可
        本方法就是通过已有材料训练得出这几个参数值
    trainMatrix：把setOfWords2Vec当中产生的数字向量都放到一个列表中形成的矩阵
    trainCategory：文档类别列表——方法loadDataSet第二个返回值
'''

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文档比例，反着看就是非侮辱性文档比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):        
        if trainCategory[i] == 1:
            # 汇总侮辱性分类情况下的各词汇总数
            p1Num += trainMatrix[i]
            # 总计侮辱性分类情况下的词汇总量
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 侮辱性情况下，各词汇出现的概率——p(w | c_1)，仍旧以向量形式展现
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive









