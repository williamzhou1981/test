# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 18:09:15 2018

@author: Administrator
"""

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1],[1.0, 1.0],[0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #shape是矩阵的维度列表
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #矩阵里面所有的元素相减
    sqDiffMat = diffMat ** 2 #矩阵里面所有的元素平方
    sqDistances = sqDiffMat.sum(axis = 1) #矩阵里面所有的元素按行进行sum，形成一个只有一列的矩阵
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() #将distances中元素从小到大排序，并返回各元素原来位置的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #累加某个label被匹配上的次数 
    '''对字典内数据进行排序，参考sorted函数和operator模块的说明
        都是为了方便进行数值操作提供的工具
        另，python 3.2之后dict类不再具有iteritems这个属性，被items属性替代
        原文的代码是python2版本的，这里需要做这个修改才能运行
        否则报异常：'dict' object has no attribute 'iteritems'
    '''
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

'''
    根据输入的文件名，从文件中获取数据转化为矩阵
'''
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        '''
            示例代码是int(listFromLine[-1])，画蛇添足了
            这个classLabelVector只需要存储状态就可以了，没必要一定是数值
            同时后续的例子当中采用的是字符串描述的，此处的int还可能引发异常
            当然也可以麻烦一点：把字符串转成16进制，然后把16进制转成10进制
            这里简单化处理，去掉int，目前看来没有问题
        '''
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector



'''
    归一化数据
    就是所有的数据都归一到0-1这个范围
'''
def autoNorm(dataSet): # dataSet is a matrix
    minValues = dataSet.min(0) # get the min value by column
    maxValues = dataSet.max(0) # get the max value by column
    ranges = maxValues - minValues
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minValues, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minValues

'''
    用10%的数据进行测试
'''
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        '''
            normMat[i,:] is the input to test
            the other args are the elements in the training set
        '''
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        #there's no print operator but function in python 3.X
        '''
            原示例代码此处是%d占位符
            不正确，因为datingTestSet.txt文件当中的Labels是字符串不是字符
        '''
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f", (errorCount / float(numTestVecs)))
    
'''
    约会网站预测函数
'''    
def classifyPerson():
    resultList = ['didntLike', 'smallDoses', 'largeDoses']
    percentTats = float(input("percentage of time spent playing video games?"))
    # there's no raw_input but input in python 3
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    '''
        在datingTestSet2文件中Label是用数字格式来表述的
        数字表述的1,2,3正好和resultList里面的下标对应
        因此才可以有如下的写法
        但是注意即使是数字格式，如果直接减去number类型也会报错，必须加上int强制转换类型
        这点和原示例代码不一致
    '''
    print(classifierResult)
    print("You will probably like this person: ", resultList[int(classifierResult)-1])
    
'''
    把图片转化为向量
    这个方法仅对本章节附带的特殊格式存储的图片有效
'''    
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

'''
    识别手写数字
'''
from os import listdir
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat,hwLabels,3)
        print("The classifier came back with: %d, the real answer is: %d" %
               (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))





    
    
    
