from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

#决策树桩的分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))#创建一个列向量，全为1，表示返回的类标签暂时为1
    if threshIneq == 'lt':#当要求左边全为负类，即-1时
        retArray[dataMatrix[:,dimen] <= threshVal] = -1#第dimen类小于threshval都变为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray

#创建决策树桩
#dataArr为输入数据，classLabels为类标签（列向量），D为每个样本的权重（类型为mat）
#返回最佳决策树桩，为字典，和最小误差，以及 以最好条件判断的结果
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr) ; labelMat = mat(classLabels).T#矩阵化
    m,n = shape(dataMatrix)
    numSteps = 10.0 #设置步进次数
    bestStump = {} ; bestClasEst = mat(ones((m,1)))#错误率最小时的判断结果
    minError = inf
    for i in range(n):#循环所有特征
        rangeMin = dataMatrix[:,i].min() ; rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):#循环每个步长，这里用j*stepSize来表示步长，
                                            # 其中步长可以超出范围，因为可以认为全部都为正类或者负类
            for inequal in ['lt','rt']:#用于表示大于还是小于阈值为-1
                threshVal = rangeMin + j * stepSize#用步长来表示阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))#创建一个与labelMat同等大小的mat，全为1
                errArr[predictedVals == labelMat] = 0#将预测正确的值设置为0
                weightedError = D.T * errArr#计算带权重的错误率
                # print('split:dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f' % \
                #       (i,threshVal,inequal,weightedError))
                if weightedError < minError:#记录权重错误率最小的情况
                    minError = weightedError
                    bestClasEst = predictedVals.copy()#错误率最小时的判断结果
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

#构建adaboost
#输入的是训练集，类标签和迭代次数
#weakClassArr返回adaboost分类器，用list表示，里面由字典组成，每个字典包括一个基学习器
#aggClassEst是f(x)最终判断的结果，利用sign(aggClassEst)可以判断类别
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []#包含每个弱分类器的判断条件
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)#表示每个样本的权重，初始化时都相等
    aggClassEst = mat(zeros((m,1)))#表示f(x)=sum(alpha_m * G_m(x))，即综合所有弱学习器判断的结果，书上叫“累计类别估计值”
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr,classLabels,D)#构建决策树桩
        # print('D:',D.T)
        alpha = float(0.5 * log((1-error)/max(error,1e-16)))#求基学习器的权重，这里的max是防止error为0，
                                                            # 这里要加float，不然就是一个列表，否则就要改下面expon
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('classEst:',classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst#将此次运算的alpha*G(x)加上
        # print('aggClassEst:',sign(aggClassEst.T))
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T , ones((m,1)))#预测不正确的值为1，预测正确的值为0，
                                                                                   # 最后相乘得出的结果是预测错误的矩阵
        errorRate = aggErrors.sum()/m#计算错误率
        # print('total error: ', errorRate, '\n')
        if errorRate == 0.0:
            break#当错误率为0时，跳出循环，不在执行
    return weakClassArr, aggClassEst

#根据adaboost来判断datToClass属于哪一类，datToClass可以是一个或多个待分类样例
#classifierArr为adaboost分类器
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))#表示f(x)，每个基学习器的线性叠加
    for i in range(len(classifierArr)):#循环每个基分类器
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#根据基学习器，预测当前数据的类结果
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return sign(aggClassEst)


datMat,classLabels = loadSimpData()
D = mat(ones((shape(datMat)[0],1))/5)
weakClassArr, aggClassEst = adaBoostTrainDS(datMat,classLabels,9)
print(adaClassify([5,5],weakClassArr))
# bestStump, minError, bestClasEst = buildStump(datMat,classLabels,D)
# print(bestStump)
