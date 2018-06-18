from numpy import  *
#此文件对应感知机


#创建数据
def creatData():
    DataMat = mat([[3,3],[5,4],[1,1],[0,0],[2,1]])
    LabelMat = mat([1,1,-1,-1,1])
    return DataMat,LabelMat


#w为权重，行向量，data为输入数据，行向量
def ganzhiji(data,label,w,b):
    mark = 1 #1表示有更新，0表示无更新
    if label*(w*data.T+b)<=0:
        w += label*data
        b += label
        mark = 1
    else:
        mark = 0
    return w,b,mark

#主要函数，返回超平面对应的w和b
def test():
    DataMat,LabelMat = creatData()
    w=zeros((1,shape(DataMat)[1]))
    b=0
    n_time = int(shape(DataMat)[0])
    i_time = 0
    while i_time < n_time:
        w,b,mark=ganzhiji(DataMat[i_time,:],LabelMat[0,i_time],w,b)
        if mark == 0:
            i_time +=1
        else:
            i_time = 0
    return w,b

def picture():
    import matplotlib.pyplot as plt
    DataMat, LabelMat = creatData()
    DataArr = array(DataMat)
    w,b=test()
    positiveCasesX=[];positiveCasesY=[]
    negativeCasesX=[];negativeCasesY=[]
    for i in range(shape(LabelMat)[1]):
        if LabelMat[0,i] == 1:#正例
            positiveCasesX.append(DataArr[i][0])
            positiveCasesY.append(DataArr[i][1])
        else :
            negativeCasesX.append(DataArr[i][0])
            negativeCasesY.append(DataArr[i][1])
    plt.figure()
    plt.plot(positiveCasesX,positiveCasesY,'ro')
    plt.plot(negativeCasesX,negativeCasesY,'go')
    x=arange(0,4,0.1)
    y=-(w[0,0]*x+b)/w[0,1]
    plt.plot(x,y)
    print(w,b)
    plt.show()

#与picture()函数的作用一样，只是上面用list，而这里是array
def picture1():
    import matplotlib.pyplot as plt
    DataMat, LabelMat = creatData()
    DataArr = array(DataMat)
    w,b=test()
    positiveCases=[]
    negativeCases=[]
    for i in range(shape(LabelMat)[1]):
        if LabelMat[0,i] == 1:#正例
            positiveCases.append(DataArr[i])
        else :
            negativeCases.append(DataArr[i])
    positiveCasesArr=array(positiveCases)
    negativeCasesArr=array(negativeCases)
    plt.figure()
    plt.plot(positiveCasesArr[:,0],positiveCasesArr[:,1],'ro')
    plt.plot(negativeCasesArr[:,0],negativeCasesArr[:,1],'go')
    x=arange(0,4,0.1)
    print(type(w))
    y=-(w[0,0]*x+b)/w[0,1]
    plt.plot(x,y)
    print(type(w))
    print(w,b)
    plt.show()

