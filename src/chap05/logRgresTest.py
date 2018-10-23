# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:24:13 2018

@author: Administrator
"""


import logRgres
# test load dataset
#print(logRgres.loadDataSet())
###############################

# test bestfit by gradAscent
#dataArr, labelMat = logRgres.loadDataSet()
#weights = logRgres.gradAscent(dataArr, labelMat)
#logRgres.plotBestFit(weights.getA()) # weights.getA() 把矩阵转化为数组

#test stocGradAscent
#dataArr, labelMat = logRgres.loadDataSet()
#weights = logRgres.stocGradAscent0(dataArr, labelMat)
'''
    此处因为在stocGradAscent当中已经有把矩阵转化为数组的处理了
    不需要weights.getA()这样的转化操作，多了这个操作反而会出现异常如下：
        AttributeError: 'numpy.ndarray' object has no attribute 'getA'
'''
#logRgres.plotBestFit(weights)

# test stocGradAscent with random alpha
dataArr, labelMat = logRgres.loadDataSet()
weights = logRgres.stocGradAscent1(dataArr, labelMat)
'''
    此处因为在stocGradAscent当中已经有把矩阵转化为数组的处理了
    不需要weights.getA()这样的转化操作，多了这个操作反而会出现异常如下：
        AttributeError: 'numpy.ndarray' object has no attribute 'getA'
'''
logRgres.plotBestFit(weights)