from numpy import *
###此文档主要构建CART算法的回归树和模型树，而基尼指数类似于ID3和C4.5，在这里不累赘了

#X和Y都是连续值
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = list(map(float,line.strip().split('\t')))
        dataMat.append(curline)
    return dataMat

#由于使用二分法，所以是一个dataset切分成2个
#dataset是mat类型
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    #注意，不能使用dataSet[dataSet[:,feature] > value]
    #因为dataSet是mat类型，如果dataset是array类型则可以
    return mat0,mat1

#回归树
#返回均值作为叶节点
def regLeaf(dataSet):
    return mean(dataSet[:,-1])

#用平方误差作为判断标准，采用“方差*数据样本数”的计算方法
def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#返回最佳划分属性和划分点
#根据叶节点和误差来进行选择
def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1):#到n-1是因为这些是特征
        for splitVal in set((dataSet[:,featIndex].T.tolist())[0]):#循环当前特征列的所有值
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)#计算划分后的误差
            if newS < bestS:#记录最小误差
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0,mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex,bestValue

#dataset是mat类型
#这个是构建CART树的算法，根据输入的参数不同，可以转变为回归树或者模型树
#其中ops[0]是可容忍误差，ops[1]是一个叶节点中最少的样本数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    #选择最佳划分属性和划分点
    feat,val = chooseBestSplit(dataSet, leafType, errType, ops)
    #判断最佳划分特征是否是空的
    #结合chooseBestSplit理解，当chooseBestSplit返回None, leafType(dataSet)，说明已经是叶节点
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat#最佳划分特征
    retTree['spVal'] = val#最佳划分点
    lSet,rSet = binSplitDataSet(dataSet,feat,val)#划分成左子树和右子树
    retTree['left'] = createTree(lSet,leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

#下面是构造模型树
#与第8章的一致，线性回归方程
def linearSolve(dataSet):
    m,n = shape(dataSet)
    X = mat(ones((m,n)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = mat(dataSet[:, -1])
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        NameError('This matrix is singular, cannot do inverse,try increasing the second value of ops')
    ws = xTx.I * X.T * Y
    return ws,X,Y

#模型树的叶节点，返回最佳回归系数
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

#模型树的误差，平方和表示
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

#下面是根据测试集来判断
#下面两个函数是为了配合treeForeCast，表示使用的是哪种模型
#回归树
def regTreeEval(model,inDat):
    return float(model)

#模型树
def modelTreeEval(model,inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1] = inDat
    return float(X*model)

#对于输入的单个数据点，treeForeCast返回一个预测值
def treeForeCast(tree,inData,modelEval=regTreeEval):
    if not isTree(tree):#当迭代到叶节点，就返回预测值
        return modelEval(tree,inData)
    if inData[tree['spInd']] > tree['spVal']:#进入左子树
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modelEval)
        else:
            return modelEval(tree['right'],inData)

#对数据进行树结构建模，即统计N的样本的情况
def createForeCast(tree,testData,modelEval=regTreeEval):
    m = shape(testData)[0]#样本数
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat


#判断是否是叶节点，当不是叶节点时，返回0
def isTree(obj):
    return (type(obj).__name__=='dict')

import matplotlib.pyplot as plt
data = mat(loadDataSet(r'C:\Users\Administrator\Desktop\aa.txt'))
myTree = createTree(data,modelLeaf,modelErr,(0.00001,2))
plt.figure()
plt.plot(data[:,0],data[:,1],'o')
x1=arange(0.45,2.3,0.01)
y1 = -0.01615306 + 0.06537468*x1
plt.plot(x1,y1)
x2 =arange(0.0,0.45,0.01)
y2 = 0.0307289942*x2
plt.plot(x2,y2)
ws,a,b=linearSolve(data)
x3 = arange(0.0,2.3,0.01)
y3 = -0.00733787 + 0.05985607*x3
plt.plot(x3,y3)
plt.show()
print(ws)