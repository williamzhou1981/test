# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:36:17 2018

@author: Administrator
"""

import trees
fr = open("lenses.txt")
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses,lensesLabels)
print(lensesTree)

'''
    自己增加的测试分类的代码
'''
testVec = ['young', 'myope','no','normal']
#lenseLabel = trees.classify(lensesTree, lensesLabels,testVec)
print(trees.classify(lensesTree, lensesLabels,testVec))