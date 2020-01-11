from numpy import *

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):#100*1
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):#dataMatIn,list类型:[[x,x,x...,x],[y,y,y,y....],...]  classLabels:[1,0,1,1,0....,1]
    dataMatrix=mat(dataMatIn) # 100*3
    labelMat=mat(classLabels).transpose()# 100*1
    m,n=shape(dataMatrix) #100,3
    alpha=0.001
    maxCycle=500
    weights=ones((n,1)) #3*1
    for k in range(maxCycle):
        v1=dataMatrix*weights#100*1
        h=sigmoid(v1)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error #transpose转秩:
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights=wei.getA() #matrix转为多维数组numpy.ndarray
    dataMat,labelMat=loadDataSet()
    dadaArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i][1])
            ycord1.append(dataArr[i][2])
        else:
            xcord2.append(dataArr[i][1])
            ycord2.append(dataArr[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]#w0+w1*x1+w2*x2
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscents(dataMatrixIn,classLabels):#dataMatrixIn:list 随机梯度下降
    m,n=shape(dataMatrixIn)
    dataMatrix=array(dataMatrixIn) #nsarray
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        sum1=sum(dataMatrix[i]*weights)
        h=sigmoid(sum1)
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrixIn,classLabels,numIter=150):#改进的随机梯度下降
    m,n=shape(dataMatrixIn)
    dataMatrix=array(dataMatrixIn)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            alaph=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alaph*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])#del是list的列表操作，删除一个或者连续几个元素
    return weights

def test_logistic():
    dataArr,labelMat=loadDataSet()
    w=stocGradAscents(dataArr,labelMat)#返回matrix
    w=stocGradAscent1(dataArr,labelMat)
    print(w)


def classifyVector(inX,weight):
    prob=sigmoid(sum(inX*weight))
    if prob>0.5:return 1.0
    else:return 0.0

def colicTest():
    frTrain=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch05/horseColicTraining.txt')
    frTest=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch05/horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[2]))
    traingWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)

    for line in frTest.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        c=classifyVector(array(lineArr),traingWeights)
        print(c)

colicTest()






















