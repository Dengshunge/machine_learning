from numpy import *

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#kNN主要算法，inX为输入行向量，dataSet是训练集特征，labels是训练集结果
def classify0(inX,dataSet,labels,k):
    m = shape(dataSet)[0]
    diffMat = tile(inX,(m,1)) - dataSet#tile函数是将向量转变成矩阵，而这一步的含义是取得与每个点的误差
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()#排序，返回原先的下标，这里是升序排列
    classCount = {}  # 创建一个空字典
    for i in range(k):
        voteIlaber = labels[sortedDistIndicies[i]]
        classCount[voteIlaber] = classCount.get(voteIlaber,0) + 1
    sortedClassCount = sorted(classCount.items(),key = lambda x:x[1],reverse=True)#item会将字典的内容转换为list
                                             #利用lambda来排序，这里记得要加上key
    return sortedClassCount[0][0]

group,labels = createDataSet()
print(classify0([1,0],group,labels,3))