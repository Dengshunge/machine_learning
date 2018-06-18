from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

#计算俩向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB,2)))

#随机产生k个质心
def randCent(dataSet, k):
    dataArr = array(dataSet)
    n = shape(dataArr)[1]
    centroids = mat(zeros((k,n)))#用于返回k个随机质心的mat
    for j in range(n):
        minJ = dataArr[:,j].min()#提取第j列的最小值
        rangeJ = float(dataArr[:,j].max() - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))#在rangeJ范围内，生成一个1列array，并加入到mat中
    return centroids

#利用kMeans进行聚类
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]#样例数
    clusterAssment = mat(zeros((m,2)))#创建一个簇分配结果矩阵，第0列记录输入哪一个簇，第1列记录存储误差
    centroids = createCent(dataSet,k)#随机产生k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#循环每个样例，分配其属于哪一簇
            minDist = inf ; minIndex = -1
            for j in range(k):#分配质心
                distIJ = distMeas(centroids[j,:],dataSet[i,:])#计算第i个样例距离j质心的距离
                if distIJ < minDist:
                    minDist = distIJ
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True#此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代
            #记录该样例的信息，包括属于簇和误差
            clusterAssment[i,:] = minIndex,minDist**2
            # print(centroids)
        #重新计算质心
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]#这里这里，如果用到这种方法，需要转换为array
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

#此为二分kMeans
#返回质心信息和簇分配结果矩阵
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))#创建一个簇分配结果矩阵，第0列记录输入哪一个簇，第1列记录存储误差
    #创建一个初始簇，所有样例都归为1个簇
    centroid0 = mean(dataSet,axis=0).tolist()[0]#质心
    centList = [centroid0]#用于保存质心信息
    for j in range(m):#当所有样例都归为一个簇时，clusterAssment第0列为0，第1列为质心到每个样例的距离的平方(即误差)
        clusterAssment[j,1] = distMeas(mat(centroid0),dataSet[j,:])**2
    #循环产生k个簇
    while len(centList) < k:
        lowestSSE = inf#误差无穷大，根据误差来选择划分
        for i in range(len(centList)):#对现行的簇进行处理
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0],:]#获取属于第i类簇的所有样例
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster,2,distMeas)#对属于第i类簇的所有样例进行2分聚类
            sseSplit = sum(splitClustAss[:,1])#计算划分后的误差
            #计算划分前不属于第i类簇的样例的误差，注意是不属于
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            # print ("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:#因为肯定需要划分的，所以选择误差下降最大的那簇
                bestCentToSplit = i#应该划分原来的哪一簇
                bestNewCents = centroidMat#划分后新的2个质点
                bestClustAss = splitClustAss.copy()#划分后的簇分配结果矩阵
                lowestSSE = sseSplit + sseNotSplit
        #更新簇类别和误差
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)#将划分的属于第1簇的编号变更为centList的长度，即放在最后
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit#将划分的属于第1簇的编号变更为原来划分的编号
        # print ('the bestCentToSplit is: ',bestCentToSplit)
        # print ('the len of bestClustAss is: ', len(bestClustAss))
        #将质心信息加入centList
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]#对应于上面的bestClustAss[:,0].A == 0，
                                                     # 因为使已经存在bestCentToSplit的编号，所以可以是直接替换
        centList.append(bestNewCents[1, :].tolist()[0])#将新的信息添加到后面
        #更新簇分配结果矩阵，因为bestClustAss已经经过修改，所以只要把属于第bestCentToSplit簇的信息替换掉即可
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList),clusterAssment




def picture():
    import matplotlib.pyplot as plt
    datMat = mat(loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码\machinelearninginaction\Ch10\testSet2.txt'))
    centroids, clusterAssment = biKmeans(datMat, 3)
    print(centroids)
    plt.figure()
    plt.plot(datMat[:,0],datMat[:,1],'o')
    plt.plot(centroids[:,0],centroids[:,1],'r+')
    plt.show()

# datMat = mat(loadDataSet(r'C:\Users\Administrator\Desktop\机器学习实战_源代码\machinelearninginaction\Ch10\testSet.txt'))
# biKmeans(datMat,2)
picture()