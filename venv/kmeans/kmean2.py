from numpy import *
def loadData(path):
    datalist=[]
    fr=open(path)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        fitline=list(map(float,curline))
        datalist.append(fitline)
    return datalist

def randCent(dataSet,k):
    n=dataSet.shape[1]
    centids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangJ=float(max(dataSet[:,j])-minJ)
        centids[:,j]=minJ+rangJ*random.rand(k,1)
    return centids



datalist=loadData('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch10/testSet.txt')
dataSet=mat(datalist)




