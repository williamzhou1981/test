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
dataArr, labelMat = logRgres.loadDataSet()
weights = logRgres.gradAscent(dataArr, labelMat)
logRgres.plotBestFit(weights.getA()) # 把矩阵转化为数组