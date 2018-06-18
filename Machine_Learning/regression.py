from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准线性回归，返回ws
def standRegres(xArr,yArr):
    xMat = mat(xArr);yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * xMat.T * yMat
    return ws

#局部加权线性回归，这里是直接测试了，需要输入点来求核
#这里的参数k，决定了对附近的点赋予多大的权重
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m,n = shape(xMat)
    weights = mat(eye(m))#这个是核函数权重，用来给每个数据点赋予权重
    for j in range(m):
        diffMat = xMat[j] - testPoint#这里是标量
        weights[j,j] = exp((diffMat * diffMat.T)/(-2*k**2))#最后的结果是只有对角线有值，其他位置都为0
    xTx = xMat.T * weights * xMat
    if linalg.det(xTx) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * xMat.T * weights * yMat
    return testPoint * ws

#这是误差函数，求平方误差
#要求输出进来的类型是array
def rssError(yArr,yHatArr):
    assert (type(yArr) is type(yHatArr)) and isinstance(yArr,ndarray)
    return ((yArr - yHatArr)**2).sum()

#这个函数用于标准化，每个值减去平均值并除以方差
#标准化与归一化有些许差别
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#这是岭回归
#这里注意一下，因为要配合ridgeTest，所以不需要矩阵化，主要是因为y，因为在test中已经将其转置了
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    print(shape(yMat))
    if linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * xMat.T * yMat
    return ws

#岭回归的测试函数，用于判断不用的lam值的影响
def ridgeTest(xArr,yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = yMat.mean(axis=0)#这里的结果是标量，即一个柱子
    yMat -= yMean#利用一个mat类型来减去一个标量，对应的结果是每个值都减去标量
    xMat = regularize(xMat)#标准化
    numTestPts = 30
    wmat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wmat[i,:] = ws.T
    return wmat

#前向逐步线性回归，也是缩减系数的一种方法
#eps是每次改变的量
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    #矩阵话
    xMat = mat(xArr)
    yMat = mat(yArr).T
    #标准化
    yMean = yMat.mean(axis=0)#这里的结果是标量，即一个柱子
    yMat -= yMean#利用一个mat类型来减去一个标量，对应的结果是每个值都减去标量
    xMat = regularize(xMat)#标准化
    m,n = shape(xMat)
    returnMat = zeros((numIt, n))#这是返回权重矩阵，记录每次权重发生的变化
    ws = zeros((n, 1))#权重
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        lowestError = inf#将误差设置为无穷大
        for j in range(n):#循环每个特征
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign#由于一减一加，是这层for循环后，不会影响ws的值
                yTest = xMat * wsTest
                rssE = rssError(yTest.A,yMat.A)
                if rssE <lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


xArr,yArr = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码\machinelearninginaction\Ch08\abalone.txt')
ws = standRegres(xArr,yArr)
print(stageWise(xArr,yArr))
# xMat = mat(xArr)
# yMat = mat(yArr).T
# yHat = xMat * ws
# rssError(array(yArr),yHat.A)
# print(ws)