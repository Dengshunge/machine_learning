from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]
    return mat(datArr)

#降维函数，参数是mat的data和想要降低到topNfeat维
#返回降维后的矩阵和重构数据
def pca(dataMat,topNfeat=9999999):
    #去均值，使数据变为零均值
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat-meanVals#减去均值，每行都进行相减，因为是mat
    covMat = cov(meanRemoved,rowvar=0)#计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))#计算特征值和特征向量，两者一一对应，特征向量正交
    eigValInd = argsort(eigVals)#对特征值排序，从小到大，返回的是index
    eigValInd = eigValInd[:-(topNfeat+1):-1]#逆序取得特征值最大的元素，并且去除topNfeat维以外的信息
    redEigVects = eigVects[:,eigValInd]#根据特征值来变换特征向量，使其一一对应，最大特征值对应最大的特征向量
    lowDDataMat = meanRemoved * redEigVects#计算降维后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals#重构原始数据
    return lowDDataMat,reconMat

#函数的作用是根据percentage来确定k，k为想要降到的维数
#eigVals是特征值组成的向量
#返回k
def percentage2n(eigVals,percentage):
    sort_eigVals = sorted(eigVals,reverse=1)#对特征值进行排序，从大到小
    eigVals_Sum = sum(sort_eigVals)
    tmpSum = 0
    num = 0
    for i in sort_eigVals:
        tmpSum += i
        num += 1
        if tmpSum >= (eigVals_Sum * percentage):
            return num
    return None


#处理数据，将数据中的nan值替换成列的均值
def replaceNanWithMean():
    datMat = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码\machinelearninginaction\Ch13\secom.data',' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])#计算非nan的均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

def picture():
    import matplotlib.pyplot as plt
    dataMat = loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码\machinelearninginaction\Ch13\testSet.txt')
    plt.figure()
    plt.plot(dataMat[:,0],dataMat[:,1],'o')
    lowDDataMat, reconMat = pca(dataMat,1)
    plt.plot(reconMat[:,0],reconMat[:,1],'+')
    plt.show()

#此函数的目的是画出方差百分比
def test():
    datMat = replaceNanWithMean()
    meanVals = mean(datMat,axis=0)
    meanRemoved = datMat-meanVals#减去均值，每行都进行相减，因为是mat
    covMat = cov(meanRemoved,rowvar=0)#计算协方差
    eigVals, eigVects = linalg.eig(mat(covMat))#计算特征值和特征向量，两者一一对应，特征向量正交

    sort_eigVals = sorted(eigVals,reverse=1)#对特征值进行排序，从大到小
    eigVals_Sum = sum(sort_eigVals)
    tmpSum = 0
    Arr = []
    for i in sort_eigVals:
        tmpSum += i
        temp = tmpSum/eigVals_Sum
        Arr.append(temp)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(Arr,'o',markersize = 1)
    plt.show()


# dataMat = replaceNanWithMean()
# pca(dataMat,1)
picture()
