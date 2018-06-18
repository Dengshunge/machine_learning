from numpy import *

#dataSet前两列对应的是lebels，而最后一列对应“是不是”
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

#加载西瓜书的数据
def loadDataSet():
    dataSet = []
    fr = open(r'C:\Users\Administrator\Desktop\22.txt')
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataSet.append(curLine)
    labels = ['色泽','根蒂','敲声','纹理','脐部','触感']
    return dataSet,labels

#计算香农熵
def calcShannonEnt(dataSet):
    from math import log
    numEntries = shape(dataSet)[0]#样本总数
    # numEntries = len(dataSet)  # 样本总数，与上述结果一样，len返回的是总数目树，对这里是行数
    labelCounts = {}#用于统计每个结果对应的次数，即yes和no分别对应的次数
    for feature in dataSet:
        labelCounts[feature[-1]] = labelCounts.get(feature[-1],0) + 1
    #开始计算香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries#第i类所占的比例
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#为了使用上面计算香农熵的函数，也为了方便使用下面的函数
#所以弄一个dataSet切分函数
#返回一个切分好的函数
def splitDataSet(dataSet, axis, value):
    retDataSet = []#创建一个返回列表
    #循环所有例子，当第axis类的value值相等时，提取出来，并放在retDataSet
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]#缓存函数
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#计算信息增益，返回最佳划分特征
def chooseBestFeatureToSplist(dataSet):
    baseEntropy = calcShannonEnt(dataSet)#香农熵
    numFeatures = len(dataSet[0])-1#特征数量
    bestInfoGain = 0.0#代表最好的信息增益
    bestFeature = -1#最好的划分特征
    for i in range(numFeatures):#循环所有特征
        featureList = [example[i] for example in dataSet]#提取第i类的所有特征值
        uniqueVals = set(featureList)#不重复的特征数目
        newEntropy = 0.0
        #下面的for循环是计算信息增益的第二项
        for key in uniqueVals:#循环对应的特征值
            subDataSet = splitDataSet(dataSet,i,key)#对第i个特征的key特征值进行划分
            prob = len(subDataSet)/float(len(dataSet))#计算第i个特征的key特征值在样本中所占的比例
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy#信息增益，对应第i类的
        if infoGain > bestInfoGain:#选择信息增益最大的
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#计算信息增益比，返回最佳划分特征
def chooseBestFeatureToSplist_IGR(dataSet):
    from math import log
    baseEntropy = calcShannonEnt(dataSet)#香农熵
    numFeatures = len(dataSet[0])-1#特征数量
    bestInfoGain_ratio = 0.0#代表最好的增益率
    bestFeature = -1#最好的划分特征
    for i in range(numFeatures):#循环所有特征
        featureList = [example[i] for example in dataSet]#提取第i类的所有特征值
        uniqueVals = set(featureList)#不重复的特征数目
        newEntropy = 0.0#信息增益的第二项
        IV = 0.0 #增益率的分母
        #下面的for循环是计算信息增益的第二项
        for key in uniqueVals:#循环对应的特征值
            subDataSet = splitDataSet(dataSet,i,key)#对第i个特征的key特征值进行划分
            prob = len(subDataSet)/float(len(dataSet))#计算第i个特征的key特征值在样本中所占的比例
            IV -= prob * log(prob,2)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy#信息增益，对应第i类的
        infoGain_ratio = infoGain / IV
        if infoGain_ratio > bestInfoGain_ratio:#选择增益率最大的
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i
    return bestFeature

#返回标签值最多的样本，此时输入的参数是标签值的列表，无特征值
#这个是自己写的函数，与书上有差别
#书上是一个一个统计，而我这个是利用set，一次性统计
#而且这里使用lambda表达式！
def majorityCnt(classList):
    uniquedata = set(classList)
    temp = {}#用于储存每个标签对应的个数
    for key in uniquedata:
        temp[key] = data.count(key)
    sortedtemp = sorted(temp.items(),key = lambda x:x[1],reverse = True)#返回一个排好序的list，这个语句值得好好研究！
    return sortedtemp[0][0]

def createTree(dataSet,labels_test):
    labels = labels_test[:]
    classList = [example[-1] for example in dataSet]#将最后的标签值提取出来
    if classList.count(classList[0]) == len(classList):#判断是否都是同一类
        return classList[0]
    if shape(dataSet)[1] == 1:#判断是否剩下0个特征
        return majorityCnt(classList)#返回标签值最多的样本
    #下面代码是创建决策树
    bestFeat = chooseBestFeatureToSplist(dataSet)#选出最好特征
    bestFeatLabel = labels[bestFeat]#最好特征对应的特征名字
    myTree = {bestFeatLabel:{}}#创建决策树，以字典为结构
    del labels[bestFeat]#删除最好特征对应的特征名字
    #下面两行是提取最好特征对应的特征值有多少种
    featValue = [example[bestFeat] for example in dataSet]
    uniquefeatVal = set(featValue)
    #循环最好特征对应的所有特征值，加入到树中
    for key in uniquefeatVal:
        sublabels = labels[:]
        myTree[bestFeatLabel][key] = createTree(splitDataSet(dataSet,bestFeat,key),sublabels)#新的决策树
    return myTree

#根据生成的决策树和分类标签，来判断testvec是属于哪一类的
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]#字典的key
    secondDict = inputTree[firstStr]#字典key对应的value
    featIndex = featLabels.index(firstStr)#key对应标签的index
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel =secondDict[key]
    return classLabel