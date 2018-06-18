from numpy import *

#创建数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('C:\\Users\\Administrator\\Desktop\\机器学习实战_源代码\\machinelearninginaction\\Ch05\\testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])#100*3矩阵
        labelMat.append(int(lineArr[-1]))#1*100行向量
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#梯度上升法，输入是训练集的特征，训练集的结果(行向量)
#返回的结果是最佳回归系数
def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)#100*3矩阵
    labelMat = mat(classLabels).T#100*1列向量
    m,n = shape(dataMatrix)
    weights = ones((n,1))#权重
    alpha = 0.001#步长
    maxCycles = 500#循环次数
    for i in range(maxCycles):
        hx = sigmoid(dataMatrix * weights)
        error = labelMat - hx
        weights += alpha * dataMatrix.T * error
    return weights  # 返回的类型是numpy数组

#画出分类边界
def plotBestFit(weights_shuzu):
    import matplotlib.pyplot as plt
    weights = array(weights_shuzu)#转化为array
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

#随机梯度上升法，此版本是循环了所有训练集
#利用一个样例来对权重进行更新
def stocGradAscent0(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)#矩阵话
    labelMat = mat(classLabels).T#矩阵话
    m,n = shape(dataMatrix)
    weights = ones((n,1))#权重
    alpha = 0.01
    for i in range(m):
        hx = sigmoid(sum(dataMatrix[i] * weights))
        error = float(labelMat[i] - hx)
        weights += alpha * error * dataMatrix[i].T
    return weights

#随机梯度上升法，此版本是利用一个例子来更新最佳回归系数
#随机选择样例来更新，且变动步长
def stocGradAscent1(dataMatIn,classLabels,numIter=150):
    dataMatrix = mat(dataMatIn)#矩阵话
    labelMat = mat(classLabels).T#矩阵话
    m,n = shape(dataMatrix)
    weights = ones((n,1))#权重
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))#随机选择一个数
            theChosenOne = dataIndex[randIndex]#该随机数字对应的真实index
            del dataIndex[randIndex]
            hx = sigmoid(sum(dataMatrix[theChosenOne] * weights))
            error = float(labelMat[theChosenOne] - hx)
            weights += alpha * error * dataMatrix[theChosenOne].T
    return weights

dataMat,labelMat = loadDataSet()
weights = stocGradAscent1(dataMat,labelMat)
plotBestFit(weights)
print(weights)

