# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:40:42 2018

@author: Administrator
"""


import trees
myDat, labels = trees.createDataSet()

print(labels)
# 原书例子是直接写死一颗决策树，这里还是利用了训练的过程，更真实一些
myTree = trees.createTree(myDat, labels)
print(myTree)
print(labels)
#print(list(myTree.keys())[0])
print(trees.classify(myTree, labels, [1,0]))
print(trees.classify(myTree, labels, [1,1]))