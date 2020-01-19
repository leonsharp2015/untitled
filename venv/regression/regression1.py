from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
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

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat #x存储是(x1,x2,...xk)的格式，也就是单个向量X:(x1,x2,...xk)的转秩
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat) #最小误差的w最优解(xT*x)^-1*xT*y
    return ws

xArr,yArr=loadDataSet('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch08/ex0.txt')
ws=standRegres(xArr,yArr)
print(ws)




















