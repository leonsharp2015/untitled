
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch06/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

dataArr,labelArr=loadDataSet()
print(labelArr)