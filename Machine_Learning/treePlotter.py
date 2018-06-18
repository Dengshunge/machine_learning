import matplotlib.pyplot as plt
#引入中文字体
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

#此例子树用于画决策树，可以直接调用createPlot(inTree)即可，可以当成一个库来用

decisionNode = dict(boxstyle="sawtooth", fc="0.8")#决策节点的属性。boxstyle：文本框类型；sawtooth是锯齿形
leafNode = dict(boxstyle="round4", fc="0.8")#叶子节点的属性，参数用法同上
arrow_args = dict(arrowstyle="<-")#箭头属性

#绘制带箭头的注解，this is all the graph吧
#nodeTxt:要显示的文本
#centerPt：文本的中心点
#parentPt:箭头指向文本的点
#nodeType:属性
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
#说明：annotate是注释，其中，xy是箭头尖的坐标，xytext设置注释内容显示的中心位置，
#xycoords和textcoords是坐标xy与xytext的说明（按轴坐标），若textcoords=None，则默认textcoords与xycoords相同，若都未设置，默认为data
#va/ha设置节点框中文字的位置，va为纵向取值为(u'top', u'bottom', u'center', u'baseline')，ha为横向取值为(u'center', u'right', u'left')

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #判断关键字key对应的值是不是字典，即判断是不是叶子
        if type(secondDict[key]).__name__=='dict':
            numLeafs +=getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth : maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
#if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

#def createPlot():
#    fig = plt.figure(1, facecolor='white')#创建一个画布，背景为白色
#    fig.clf()#清空画布
#   ax1是函数createPlot的一个属性，这个可以在函数里面定义也可以在函数定义后加入也可以
#   frameon表示是否绘制坐标轴矩形 ，也就是外面的大框框
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode(U'决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode(U'叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()