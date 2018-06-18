from numpy import *

class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name =nameValue
        self.count = numOccur
        self.nodeLink = None#用于连接相似的元素项
        self.parent = parentNode#父节点
        self.children = {}

    def inc(self,numOccur):
        self.count += numOccur

    def disp(self,ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def loadSimpDat1():
    simpDat = [list('ABCEFO'),list('ACG'),list('EI'),list('ACDEG'),list('ACEGL'),
               list('EJ'),list('ABCEFP'),list('ACD'),list('ACEGM'),list('ACEGN')]
    return simpDat

#对初始数据进行格式化处理
#将数据转换成字典，原始数据变为frozenset，而字典的值为1
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

#构建FP树，此时传入的dataSet是经过格式化处理的，即是一个字典
def createTree(dataSet, minSup=1):
    headerTable = {}#头指针表
    #下面是第一次遍历数据集，目的是构建头指针表，表示每个元素出现的次数及删除不满足支持度的项
    for trans in dataSet:#这里读取的是字典的关键字
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]#头指针表，此时是一个key对应一个int值
    for k in list(headerTable.keys()):#去头指针表的key
        #删除出现次数少于minsup的项，即移除不满足最小支持度的元素项
        if headerTable[k] < minSup:
            del headerTable[k]
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:#如果所有项都不频繁，就不需要进行下一步处理
        return None,None
    for k in headerTable.keys():
        headerTable[k] = [headerTable[k],None]#修改头指针表，此时修改成一个key对应一个list，包含原来的int和相似link
    #下面是第二次遍历数据集，目的是构建FP树
    retTree = treeNode('Null Set',1,None)#构建头结点
    for tranSet,count in dataSet.items():#因为dataSet是字典，所以有两个值，就需要两个对应
        localD = {}
        for item in tranSet:#针对dataSet.key的每一行的每个元素
            if item in freqItemSet:#只对频繁项集进行操作
                localD[item] = headerTable[item][0]#等号右侧表示出现的次数
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(),key = lambda p:p[1], reverse=True)]#排序好，取其关键字
            #使用排序后的频繁项集对数进行填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree,headerTable

#对排序好后的频繁项集更新来构建树
#参数分别为排序好的频繁项集，FP树，头指针表，count
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#由于items是排序好的，所以判断第一个元素项是否作为子节点存在
        inTree.children[items[0]].inc(count)#如果存在，则增加其计数值
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree)#如果不存在，则构建一个子节点
        #下面的if是用于更新头指针表，或者可以说更新节点链接
        #当头指针表headerTable[items[0]][1]为None时，说明第一次加入，直接修改头指针表即可
        if headerTable[items[0]][1] == None:#更新头指针表，指向头结点下的items[0]
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:#当不为None时，说明已经有一条节点链表，将新产生的节点加入到链表中
            #此时由于这个算法是迭代的，inTree.children是新生成的节点，其中inTree不是一开始的父节点，而是新生成的节点的父节点
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#当items不多余1个元素时，表明仍有未分配完的树，迭代次算法，在items[0]下继续构建树
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

# 对节点链接进行更新，由于targetNode新产生的节点，为加入原先的链表中，所以循环nodeToTest链表到链尾，
# 然后将链尾的节点链接修改成targetNode，即将targetNode加入到链表中
def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# 迭代上溯整棵树，利用parent
# leafNode为一个字节，而prefixPath是list，表明路径
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

#从头指针表中的指针开始，一直迭代到链表的末端
#basePat表明要读取的元素，在代码中没用到
#treeNode表示头指针中某个元素的链表头
def findPrefixPath(basePat, treeNode):
    condPats = {}#用于保存条件模式基，及其对应的计数值关联
    while treeNode != None:#循环值链尾
        prefixPath = []#表示条件模式基
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count#第0个元素是当前的元素，即basePat，后面的才是条件模式基
        treeNode = treeNode.nodeLink
    return condPats

#此函数为挖掘FP树，输入参数分别是FP树，头指针表，最小支持度，空的set（最终不输出），最终的频繁项集（函数结束后会为非空）
# 函数mineTree对参数inTree代表的FP树进行频繁项集挖掘。首先对headerTable中出现的单个元素按出现频率从小到大排序，
# 之后将每个元素的条件模式基作为输入数据，建立针对当前元素的条件树，如果生成的这棵条件树仍有元素，就在这棵条件树里寻找频繁项集，
# 因为prefix参数是在递归过程中不断向下传递的，因此由最初的headerTable中的某个元素x衍生出的所有频繁项集都带有x。
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p:p[1][0])]#从小到大排序，提取第0项，表明从低端开始
    for basePat in bigL:#从底层开始
        #加入频繁项集列表
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print('finalFrequent Item: ', newFreqSet)
        freqItemList.append(newFreqSet)#将频繁项添加进list中
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])#得到条件模式基
        # print('condPattBases :', basePat, condPattBases)
        #利用条件模式基来构建FP树
        myCondTree,myHead = createTree(condPattBases,minSup)
        # 将创建的条件基作为新的数据集添加到fp-tree
        # print('head from conditional tree: ', myHead)
        if myHead != None:
            # print('conditional tree for: ',newFreqSet)
            # myCondTree.disp(1)
            mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)

simpDat = loadSimpDat1()
initSet = createInitSet(simpDat)
# print(initSet)
myFPtree, myHeaderTab = createTree(initSet,2)
myFPtree.disp(1)
# print(myHeaderTab)
freqItems = []
mineTree(myFPtree,myHeaderTab,2,set([]),freqItems)
