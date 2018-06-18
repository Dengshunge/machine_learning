from numpy import *

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData1():
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

#下面三种相似度，输入都是列向量
#欧式距离相似度
def euclidSim(inA,inB):
    return 1.0/(1.0+linalg.norm(inA-inB))

#皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA)<3:
        return 1.0 #为什么
    return 0.5+0.5*corrcoef(inA,inB,rowvar = 0)[0][1]

#余弦相似度
def cosSim(inA,inB):
    num = inA.T * inB
    denom = linalg.norm(inA)*linalg.norm(inB)
    return 0.5+0.5*(num/denom)

#用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
#输入参数分别是数据矩阵(包含多个用户对多个菜式的评分)，用户编号，相似度计算方法，物品编号(未尝试过的菜式)
#列表示物品，行表示用户
#循环每个物品，都与该未尝试过的菜式做相似度计算，当这两列的相似度高时，就认为该用户也会喜欢这个菜式，就推荐
#但实际中推荐不是简单的比较相似度，而是将相似度相加，并乘以评分等级，最后两者相除，返回预测评分等级
def standEst(dataMat,user,simMeas,item):
    n = shape(dataMat)[1]#获取物品数目
    simTotal = 0.0#相似度
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:#跳过值为0的项，值为0表示用户未尝试过此菜式
            continue
        #overLap是给出两列都已经有评分的项
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        # overLap1 = nonzero((dataMat[:, item].A > 0) & (dataMat[:, j].A > 0))[0]#效果与上面一样
        if len(overLap)==0:#若无重叠，说明相似度为0
            similarity = 0
        else:
            #当有重叠的时候，计算两列的相似度
            similarity = simMeas(dataMat[overLap,j],dataMat[overLap,item])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        #不是很懂为什么要这么计算？
        simTotal += similarity#相似度不断叠加
        ratSimTotal += similarity * userRating#并考虑相似度和当前用户的评分成绩
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal#归一化，返回评分等级

#利用SVD对数据进行压缩并对物品进行评分预测
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,sigma,VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4)*sigma[:4])#得到一个四维的对角矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I#对数据进行降维，但为什么是这样呢？
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:#为什么会判断j==item，但要不要j判断结构都一样
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)#为什么这里会要求转置
        # print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal#归一化，返回评分等级



#推荐系统，根据数据集来推荐用户未尝试的菜式
#输入的参数有mat类型的数据集，用户编号，推荐个数，相似度算法，预测评分算法
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]#提取用户未尝试过的菜式
    if len(unratedItems)==0:
        return 'you rated everything'
    itemScores = []#保存菜式编号和预测评分
    for item in unratedItems:#循坏每个未尝试的菜式
        estimatedScore = estMethod(dataMat,user,simMeas,item)#预测评分等级
        itemScores.append((item,estimatedScore))
    return sorted(itemScores,key = lambda x:x[1],reverse=True)[:N]#根据评分等级排序，从大到小，返回前N个

#基于自己的理解写的
def mysvdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,sigma,VT = linalg.svd(dataMat)
    Sig4 = mat(eye(4)*sigma[:4])#得到一个四维的对角矩阵
    # xformedItems = dataMat.T * U[:,:4] * Sig4.I#对数据进行降维，但为什么是这样呢？
    xformedItems = Sig4.I * VT[:4,:]#这个对应的是压缩后的数据
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:#为什么会判断j==item
            continue
        # similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)#为什么这里会要求转置
        similarity = simMeas(xformedItems[:,item], xformedItems[:,j])#因为是行压缩，所以列不变
        # print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal#归一化，返回评分等级


data = mat(loadExData2())
n=2
a = recommend(data,n,estMethod= svdEst,N= 5)
b = recommend(data,n,estMethod= mysvdEst,N= 5)
print(a)
print(b)
# svdEst(data, 0, cosSim, 1)

