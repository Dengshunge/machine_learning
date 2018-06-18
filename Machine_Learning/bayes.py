from numpy import *

#创建词表矩阵和类标签向量
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#创建一个不重复（SET）的词向量，返回一个排好序的list
#表示词汇表，用于所有输入与此词汇表比较
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)#注意这里，或预算是2个set，即同类型
    return sorted(vocabSet)

#将输入进来的文本转换为向量，与词汇表进行比较，当输入出现这个词时，词汇表对应的位置置1
#vocabList是一个不重复的词条向量，相当于一本字典，而inputset是词条向量，切分一句话的单词组成
#通过for循环，将inputset中的每个单词循环一遍，若这个单词在vocabList中，则在返回矩阵中的相关位置置1
#这个函数的作用就只能表明该单词是否出现过
#返回是一个由0,1组成的向量
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)#创建一个与词汇表同等大小的行向量，元素全为0
    for words in set(inputSet):
        if words in vocabList:
            returnVec[vocabList.index(words)] = 1
        else:
            print('the word: %s in not in my Vocabulary!' % word)
    return returnVec

#这是词袋模型，表明每个单词出现的此时，而上面的setOfWords2Vec为词集模型，表明这个单词是否出现过
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#计算贝叶斯概率，输入一个经过setOfWords2Vec转换而得到的训练矩阵和标签向量
#返回p(x|c0)、p(x|c1)和p(c1)
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练集的个数，即trainMatrix的行数，是0，1组成的矩阵
    numWords = len(trainMatrix[0])  # 第一个训练集的长度，即含有多少个单词
    pAbusive = sum(trainCategory)/len(trainCategory)#算出p(c1)的概率，因为带侮辱的词汇的例子是1，所以相加的结果就是总共有多少个
    p0Num = ones(numWords); p1Num = ones(numWords)#防止下面的概率为0，所以是ones，主要这里生成的是向量，不是矩阵
    p0Denom = 2.0; p1Denom = 2.0  # 这两行是拉布拉斯平滑，这里的值是k*1，K为训练集结果的种类数目
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]#将侮辱性词汇的例子（每个例子是有0,1组成的向量，是setOfWords2Vec）对应位相加，
                                   # 表明不同例子中这个单词出现的次数，因为trainMatrix的1表明的是这个单词出现过
            p1Denom += sum(trainMatrix[i])  # 上面是每个单词出现的次数，而这里是总次数
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = log(p0Num/p0Denom)#两者相除，是每个单词出现的概率，用ln是为了防止下溢
    p1Vect = log(p1Num/p1Denom)
    return p0Vect, p1Vect, pAbusive  # 返回p(x|c0)、p(x|c1)和p(c1)

#分类标准，vec2Classify是要验证的向量，也是经过setOfWords2Vec转化
#返回属于哪一类
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    vec2Classify = array(vec2Classify)
    p0Vec = array(p0Vec) ; p1Vec = array(p1Vec)
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)#因为是ln，所以是相加，这里是vec2Classify中存在的单词的概率对应位相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#此为测试函数
#此函数有缺陷，当为未出现的词汇出现时，会产生错误
#原因在于创建词汇表时，未考虑未出现的词
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = setOfWords2Vec(myVocabList,testEntry)
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()
