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

def disEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def kmeans2(dataSet,k):
    m=dataSet.shape[0]
    dataAssment=mat(zeros((m,2)))
    centisd=randCent(dataSet,2)
    for i in range(m):
        v1=dataSet[i,:]
        min_distance=inf
        centid_index=0
        for j in range(k):
            c=centisd[j,:]
            dis=disEclud(v1,c)
            if(dis<min_distance):
                min_distance=dis
                centid_index=j
        dataAssment[i,:]=[min_distance,centid_index]
    return dataAssment


datalist=loadData('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch10/testSet.txt')
dataSet=mat(datalist)
dataAssment=kmeans2(dataSet,2)




