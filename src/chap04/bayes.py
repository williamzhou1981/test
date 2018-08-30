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
    切记：这个方法只是展示算法的实现原理，不能直接使用
        后续有个修改版的才能使用
    
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
            p(w)——没有找到求解这个数值的代码！！！！
        本方法就是通过已有材料训练得出这几个参数值
    trainMatrix：把setOfWords2Vec当中产生的数字向量都放到一个列表中形成的矩阵
    trainCategory：文档类别列表——方法loadDataSet第二个返回值
'''

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文档比例，反着看就是非侮辱性文档比例——p(c_1)
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
    '''
        侮辱性情况下，各词汇出现的概率——p(w | c_1)，仍旧以向量形式展现
        形如  p(w0 |c1)    p(w1 | c1)      p(wn | c1) 
            ([0,04166667, 0.04166667,....,0.125])
        这个向量是针对完整的词汇表的，意义也很明确
            即，侮辱性文档的情况下，每个词出现的概率
    '''
    p1Vect = p1Num / p1Denom
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


'''
    参数和方法的基本说明和trainNB0一致
    几处重要的修改见代码中说明
'''
def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文档比例，反着看就是非侮辱性文档比例——p(c_1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    
    '''
        利用贝叶斯分类器进行文档分类的时候，会计算Πp(wi | c_x)
        这样当任一个p(wi | c_x)为零的时候就会让最后乘积为零，误差很大
        因此
            1、改原有的zeros为ones
            2、修改分明的初始值为2.0
    '''
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):        
        if trainCategory[i] == 1:
            # 汇总侮辱性分类情况下的各词汇总数
            p1Num += trainMatrix[i]
            # 总计侮辱性分类情况下的词汇总量
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    '''
        侮辱性情况下，各词汇出现的概率——p(w | c_1)，仍旧以向量形式展现
        形如  p(w0 |c1)    p(w1 | c1)      p(wn | c1) 
            ([0,04166667, 0.04166667,....,0.125])
        这个向量是针对完整的词汇表的，意义也很明确
            即，侮辱性文档的情况下，每个词出现的概率
        如果按照这个向量进行返回，实际后续操作中产生的连续乘积会导致下溢出
        即，使用了过多的极小数字相乘，结果会非常小，甚至于计算机只能把它表示为零
        此时，采用对数处理的方式，ln(ab)=ln(a)+ln(b)这样的方式避免下溢出
        修改就是在原来的矩阵运算的结果上直接使用log即可
    '''
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

'''
    朴素贝叶斯分类器
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
        vec2Classify相当于是掩码，直接和对应的Vec相乘，只会留下对应的词汇的概率值
        sum即是把向量的分量值全部进行了累加
        之前对这些分量值都进行了对数化的处理，所以连加是没问题的，sum可以运行
        同样，后面半截的log(pClass_i)也是进行累加的
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #此处只有两个分类
    '''
        个人的看法
        因为只需要比较大小就可以确定分类，所以没必要进行p(w)的计算
        对于两个（多个）值，在比较大小的情况下，同时除以一个相同的值对结果不会有影响
        所以代码当中没有出现p(w)的计算
    '''
    if p1 > p0:
        return 1
    return 0

'''
    用论坛帖子的方式测试第一个朴素贝叶斯分类器
'''
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB1(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as : ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, ' classified as : ', classifyNB(thisDoc, p0V, p1V, pAb))
    
'''
    文档词袋模型
    上述的词汇总表是一个集合的模式，即只能表示某个词汇出现与否
    词汇出现的次数没有得到体现
    如果我们的判断需要引入词汇出现的次数，那么可以使用下面的词袋模型更加合适
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
    文件解析及完整的垃圾邮件测试函数
'''
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    # there are only 25 files in each folder
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        #print(i,": ", wordList)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 小插曲，第23号文件有不可识别字符，重新编辑了一下就可以用了
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        #print(i,": ", wordList)
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    '''
        书上的代码直接使用range方法，但是在这个版本的python输出结果是元组
        无法进行后续操作，程序报错
        修改为list之后程序正常运行
    '''
    trainingSet = list(range(50))
    testSet = []
    # 随机构建训练集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB1(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex] :
            errorCount += 1
    print("the error rate is: ", float(errorCount) / len(testSet))



























