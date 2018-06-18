from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = [];labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

#返回一个与i不一样的数字
def selectJrand(i,m):
    j = i
    while j == i:
        j=int(random.uniform(0,m))
    return j

#用于调整aj，防止过大或者过小
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

#简化版SMO
#C为惩罚系数，toler为容错率，maxIter为最大迭代次数
#为dataMatIn训练样本训练处分离超平面
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = mat(dataMatIn)#矩阵化
    labelMat = mat(classLabels).T#矩阵化
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))#初始化所有alpha都为0
    b = 0.0#初始化b
    iter = 0#迭代次数
    while iter < maxIter:
        alphaPairsChanged = 0#因为每次循环都会有alpha对变化，此时不能停止循环
                             #当所有alpha都没发生改变的时候，才能算一次成功的迭代
        for i in range(m):#循环所有样例
            fXi = float( multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[i,:].T) ) + b
            Ei = fXi - float(labelMat[i])
            #由于alpha大于C或者小于0都会调整为0或C
            #下面的判断条件见分析，太长了
            #判断条件是违反KKT条件的点
            if ((labelMat[i]*Ei<-toler)and alphas[i]<C)or((labelMat[i]*Ei>toler)and alphas[i]>0):#这些alpha对应支持向量
                j = selectJrand(i,m)
                fXj = float( multiply(alphas,labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                eta = dataMatrix[i,:] * dataMatrix[i,:].T + dataMatrix[j,:] * dataMatrix[j,:].T - \
                    2.0 * dataMatrix[i,:] * dataMatrix[j,:].T
                if eta<0.0:
                    print('eta error')
                    continue
                #用于保存alpha的旧值，方便更新alpha_i
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #判断H和L，用于确定alpha_j
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C,alphas[j]+alphas[i])
                if L == H:
                    print('L==H')#这种情况说明其中一个alpha=C，而另外一个=0，所以不用理
                    continue
                alphas[j] += labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)#更新alpha_j
                if (abs(alphas[j]-alphaJold)<0.00001):#这个判断是为了让alpha_j有足够大的更新
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])#更新alpha_i
                b1 = b-Ei-labelMat[i]*dataMatrix[i,:]*dataMatrix[i,:].T*(alphas[i]-alphaIold)-\
                    labelMat[j]*dataMatrix[i,:]*dataMatrix[j,:].T*(alphas[j]-alphaJold)
                b2 = b-Ej-labelMat[i]*dataMatrix[i,:]*dataMatrix[j,:].T*(alphas[i]-alphaIold)-\
                    labelMat[j]*dataMatrix[j,:]*dataMatrix[j,:].T*(alphas[j]-alphaJold)
                if (alphas[i]<C) and (alphas[i]>0):
                    b = b1
                elif (alphas[j]<C) and (alphas[j]>0):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1#说明已经有一对alpha发生改变
                print('iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1#如果没有alpha发生改变，则迭代次数加一
        else:
            iter =0#当alpha发生改变时，重新进行迭代，确保maxIter次都无alpha发生改变
        print('iteration number: %d' % iter)
    return b,alphas

#计算核函数，返回一个列向量
#输入X是一个mat矩阵，而A是一个列向量
#kTup是有2个参数，[0]是为了控制采用哪个核函数，[1]是核函数需要的参数
def kernelTrans(X,A,kTup):
    m, n = shape(X)
    K = zeros((m,1))
    if kTup[0] == 'lin':#线性内积
        K = X * A.T#即X的每行数据都与A做相乘，简单理解就是X_i*X
    elif kTup[0] == 'rbf':#径向基函数
        for i in range(m):#循环每行数据
            deltaRow = X[i, :] - A#每行数据都与A进行相减
            K[i] = deltaRow*deltaRow.T
        K = exp(K/(-2*kTup[1]**2))#然后取指数
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))#第一位为标志位，第二位才是误差
        self.K = mat(zeros((self.m,self.m)))#创建一个m*m的mat矩阵，用于存储核函数计算出来的值
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)#计算核函数

#计算alpha_k的误差
def calcEk(oS,k):
    fXk = float( multiply(oS.alphas,oS.labelMat).T * oS.K[:,k] ) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

#选择alpha_j，利用Ej和Ei的最大差值来选择，在alpha中选择
def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1, Ei]#将缓存eCache的ith设置为有效
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]#结果是第0列不为0的行号
    if len(validEcacheList)>1:
        for k in validEcacheList:
            if k == i:#防止重复，也可以不需要
                continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ek - Ei)#计算最大步长
            if deltaE > maxDeltaE:
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)#此种情况说明是第一次选择，所以随机选择
        Ej = calcEk(oS,j)
        return j,Ej

#更新误差Ek
def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]

#内循环，与上面简化版SMO类似
#此时返回1，表示经过修改；返回0，表示未修改
def innerL(i,oS):
    Ei = calcEk(oS,i)
    #由于alpha大于C或者小于0都会调整为0或C
    #下面的判断条件见分析，太长了
    #判断条件是违反KKT条件的点
    if ((oS.labelMat[i]*Ei<-oS.tol)and oS.alphas[i]<oS.C)or\
            ((oS.labelMat[i]*Ei>oS.tol)and oS.alphas[i]>0):#这些alpha对应支持向量
        j,Ej = selectJ(i,oS,Ei)
        #计算最优修改量
        eta = oS.K[i,i] + oS.K[j,j] - 2.0 * oS.K[i,j]
        if eta<0.0:
            # print('eta error')
            return 0
        #用于保存alpha的旧值，方便更新alpha_i
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        #判断H和L，用于确定alpha_j
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0,oS.alphas[j]-oS.alphas[i])
            H = min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L = max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H = min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L == H:
            # print('L==H')#这种情况说明其中一个alpha=C，而另外一个=0，所以不用理
            return 0
        oS.alphas[j] += oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)#更新alpha_j
        # updateEk(oS,j)#更新Ej误差，此处应该不对，应该用新的b来更新误差，而不应该用旧的
        if (abs(oS.alphas[j]-alphaJold)<0.00001):#这个判断是为了让alpha_j有足够大的更新
            # print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[i] * oS.labelMat[j] * (alphaJold - oS.alphas[j])#更新alpha_i
        # updateEk(oS, i)#更新Ei误差，此处应该不对，应该用新的b来更新误差，而不应该用旧的
        b1 = oS.b-Ei-oS.labelMat[i]*oS.K[i,i]*(oS.alphas[i]-alphaIold)-\
                    oS.labelMat[j]*oS.K[i,j]*(oS.alphas[j]-alphaJold)
        b2 = oS.b-Ej-oS.labelMat[i]*oS.K[i,j]*(oS.alphas[i]-alphaIold)-\
                    oS.labelMat[j]*oS.K[j,j]*(oS.alphas[j]-alphaJold)
        if (oS.alphas[i]<oS.C) and (oS.alphas[i]>0):
            oS.b = b1
        elif (oS.alphas[j]<oS.C) and (oS.alphas[j]>0):
            oS.b = b2
        else:
            oS.b = (b1+b2)/2.0
        updateEk(oS, j)#更新Ej误差
        updateEk(oS, i)#更新Ei误差
        return 1
    return 0

def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn),mat(classLabels).T,C,toler,kTup)#初始化
    iter = 0 #迭代次数
    entireSet = True#用来控制两种循环
    alphaPairsChanged = 0
    while(iter<maxIter) and ((alphaPairsChanged>0) or entireSet):
        alphaPairsChanged = 0
        if entireSet:
            #首先先在全集合进行搜索
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                # print('fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1 #无论alpha是否有修改，都会+1，而之前是只有当alphaPairsChanged=0时，才会+1
        else:
            #然后对支持向量进行搜索
            #2种搜索交替进行
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < oS.C))[0]#这里很巧妙，当两个bool类型相乘，取出为真的
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                # print('fullSet, iter: %d i:%d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b,oS.alphas

#利用alpha来计算ws，返回是一个列向量
def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(X)
    w = zeros((n,1))
    alphas_nonzeros = nonzero(alphas > 0)[0]#取出alpha不为0的行号
    for i in alphas_nonzeros:
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

#用错误率来表示判断
def classfySVM(k1=1.3):
    dataArr, labelArr = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码'
                                    r'\machinelearninginaction\Ch06\testSetRBF.txt')
    b,alphas = smoP(dataArr,labelArr,200, 0.0001, 10000, ('rbf',k1))
    dataMat = mat(dataArr); labelMat = mat(labelArr).T
    svInd = nonzero(alphas>0)[0]#取出alphas中不为0的行号
    sVs = dataMat[svInd]#alphas中不为0对应的数据，提醒一下，这里的用法，可以用mat矩阵，但却不能使用array
    labelSV = labelMat[svInd]
    #循环所有数据
    errorCount = 0
    for i in range(shape(dataMat)[0]):
        kernelEval = kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        fXi = kernelEval.T * multiply(alphas[svInd],labelSV) + b
        if sign(fXi) != sign(labelArr[i]):
            errorCount += 1
    print('error_rate= %f' % float(errorCount/shape(dataMat)[0]))

def test():
    dataMat, labelMat = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码'
                                    r'\machinelearninginaction\Ch06\testSetRBF.txt')
    b,alphas = smoP(dataMat,labelMat,0.6,0.001,40)

def picture():
    dataMat, labelMat = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码'
                                    r'\machinelearninginaction\Ch06\testSet.txt')
    pos = [];neg=[]
    for i in range(len(labelMat)):
        if labelMat[i] == 1:
            pos.append(dataMat[i])
        else :
            neg.append(dataMat[i])
    plt.figure()
    plt.plot(array(pos)[:,0],array(pos)[:,1],'ro')
    plt.plot(array(neg)[:,0], array(neg)[:,1], 's')
    b, alphas = smoP(dataMat, labelMat, 0.6, 0.001, 40, ('lin',0))
    ws = calcWs(alphas, dataMat, labelMat)
    x=arange(-1,10,0.1)
    y=(-b[0,0]-ws[0]*x)/ws[1]
    plt.plot(x,y)
    plt.show()

# classfySVM()
picture()