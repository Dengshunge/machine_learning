from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#创建C1，C1是大小为1的所有候选项集的集合，用来判断是否满足最小支持度
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])#注意，这里是加入项，不是单纯的数字
    C1.sort()
    return list(map(frozenset, C1))

#D为数据集，Ck为候选集，minSupport为最小支持度
#返回满足最小支持度的列表和每项的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    #统计候选集元素在数据集中出现的次数，放入字典中
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can,0) + 1
    numItems = len(D)#数据集的数量
    #下面是计算支持度，并将满足最小支持度的项加入retlist中
    retList = []#包含满足最小支持度的项
    supportData = {}#计算每项的支持度
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)#当满足最小支持度时，加入list中，这里也可以用append
        supportData[key] = support#为什么要将不满足最小支持度的项加入进去呢？
    return retList,supportData

#创建Ck,其中Lk为频繁集，而k是项数，表示合成后希望元素的个数
#这里的写法有些巧妙，首先因为Lk是已经排序好的，然后比较前k-2项，如果k-2项相同，则将2个合并成k项
#这样能极大缩减遍历次数
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1 = list(Lk[i])[:k-2]#因为需要分片，只有list支持分片
            L2 = list(Lk[j])[:k-2]
            L1.sort();L2.sort()#注意，这里一定需要排序
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

#Apriori主算法
def apriori(dataSet, minSupport = 0.5):
    #首先生成C1，C1中只有大小为1的元素
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData  = scanD(D, C1, minSupport)
    L = [L1]#此list为返回list，包含频繁项集
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2],k)#合并频繁项集
        Lk,supK = scanD(D,Ck,minSupport)#Lk为满足最小支持度的项集，从Ck中删除了不满足支持度的项，supK为Ck每项的支持度
        supportData.update(supK)#字典的更新用法，包含所有项的支持度
        L.append(Lk)
        k += 1
    return L,supportData

#生成关联规则的主函数，其中L是频繁项集，supportData来自于scanD，由满足最小支持度的项组成
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []#存储所有的关联规则
    for i in range(1,len(L)):#只取大于2项的set
        for freqSet in L[i]:#对于这组频繁项集中，例如L[1]=[frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})]
            H1 = [frozenset([item]) for item in freqSet]#将组合的值拆分成单个值，例如H1=[frozenset({2}), frozenset({3})]
            if i>1:
                # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                # 第一层时，后件数为1
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# 计算可信度，其中freqSet是一个频繁项集，例如frozenset({2, 3})
# H为当前这组频繁项集中的元素，对应上面的H1，例如[frozenset({2}), frozenset({3})]
# supportData来自于scanD，由满足最小支持度的项组成
# brl是由上面的函数传过来的，用于包含关联规则
# 返回满足最小可信度的list
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 针对项集中只有两个元素时，计算可信度
    prunedH = []#用于返回一个满足最小可信度要求的规则列表
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]#计算freqSet-conseq到conseq的支持度
        if conf >= minConf:
            print(freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))#加入元祖
            prunedH.append(conseq)
    return prunedH

# freqSet是一个频繁项集，例如[frozenset({2, 3, 5})]
# H为当前这组频繁项集中的元素,例如[frozenset({2}), frozenset({3}), frozenset({5})]
# supportData来自于scanD，由满足最小支持度的项组成
# brl是由上面的函数传过来的，用于包含关联规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet)) > (m+1):#频繁项集元素数目大于单个集合的元素数,尝试更多的合并
        Hmp1 = aprioriGen(H,m+1)#将其合并，由于是调用calcConf，其只能运算2项的
                                #例如Hmp1 = [frozenset({2, 3}), frozenset({2, 5}), frozenset({3, 5})]
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)#计算可信度
        if (len(Hmp1) > 1):
            # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)



dataSet = loadDataSet()
C1 = createC1(dataSet)
D = list(map(set,dataSet))
L,suppData = apriori(D,0.5)
# print(L)
rules = generateRules(L,suppData,0.5)
print(rules)
