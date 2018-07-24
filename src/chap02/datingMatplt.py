# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 17:21:03 2018

@author: Administrator
"""

import matplotlib
import matplotlib.pyplot as plt
import kNN2_2

datingDataMat, datingLabels = kNN2_2.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()

