from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):#最小二乘法误差的w计算
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat #x存储是(x1,x2,...xk)的格式，也就是单个向量X:(x1,x2,...xk)的转秩
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat) #最小误差的w最优解(xT*x)^-1*xT*y
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):#局部加权 testPoint某一行[1.0, 0.067732]
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0] #数据集行数
    weights = mat(eye((m))) #对角矩阵
    for i in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[i,:]     #某一行和每一行相减 x(i)-x得到的矩阵 m*2
        weights[i,i] = exp(diffMat*diffMat.T/(-2.0*k**2))#|x(i)-x|的计算＝矩阵＊矩阵T
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws,weights

def lwlrTest(testArr,xArr,yArr,k=0.2):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat


def ridgeRegres(xMat, yMat, lam=0.2): #岭回归。w=(xTx+lamda*E)^-1*xT*y
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 0-1范围之内
    # regularize X's
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)  # 方差[0.  0.08538647]
    xMat = (xMat - xMeans) / xVar #标准化
    # xMat=nan_to_num(xMat) #标准化,如果会有分子为0的情况nan,将xMat为nan的值替换为0
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1])) #30*8
    for i in range(numTestPts):#测试30次
        ws = ridgeRegres(xMat, yMat, exp(i - 10)) #不断减少lamda
        wMat[i, :] = ws.T #回归系数矩阵
    return wMat



def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):#前向逐步回归
    xMat = mat(xArr)
    yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef

    xMat = regularize(xMat)
    m,n=shape(xMat) #4177*8
    returnMat = zeros((numIt,n)) #testing code remove
    ws = zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):#测试100次
        #print(ws.T)
        lowestError = inf;#无限大的正数
        for j in range(n):#0-7
            for sign in [-1,1]:#每次走2步,步长是eps
                wsTest = ws.copy() #8*1
                wsTest[j] += eps*sign #wsTest[j]表示该特征j对误差的影响.wsTest=[0,0.01,.....,0]T
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A) #A代表将矩阵转化为array数组类
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

# xArr,yArr=loadDataSet('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch08/ex0.txt')
# test_y,weights=lwlr(xArr[0],xArr,yArr,0.001)

xArr,yArr=loadDataSet('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch08/abalone.txt')
# ridgeweight=ridgeTest(xArr,yArr)
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.plot(ridgeweight)
# plt.show()
ws=stageWise(xArr,yArr,0.01,200)
print(ws)
















