#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines(): #for each line
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine)) #这里和书中不同 和上一章一样修改
        dataMat.append(fltLine)
    #新加
    dataMat = array(dataMat)
    dataMat = mat(dataMat)
    return dataMat


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):#簇4*2
    n = shape(dataSet)[1] #2
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) # matrix [[-5.379713]]
        rangeJ = float(max(dataSet[:,j]) - minJ)
        rnd=rangeJ * random.rand(k,1) #random.rand(k,1)是k*1的ndarray随机数组
        column_value=mat(minJ + rnd) #column_value k*1
        centroids[:,j] = column_value
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #80
    clusterAssment = mat(zeros((m,2)))#存储数据点对应的最近簇索引(0-3)、点到簇质心的距离.  数据行数*2

    centroids = createCent(dataSet, k) #randCent函数，4*2的簇
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#所有数据行数80
            minDist = inf
            minIndex = -1
            for j in range(k):#0-3
                distJI = distMeas(centroids[j,:],dataSet[i,:]) #distMeas=distEclud.centroids与所有数据点的距离
                if distJI < minDist:
                    minDist = distJI #最小距离值
                    minIndex = j # 与所有数据点距离最小距离的簇的索引（0-3）
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2 #80*2.该数据点（1-80行数据点）对应的最近簇索引[第一列]、最近距离［第二列］

        #更新质心（簇）的位置
        for cent in range(k):#0-3
            # nonzero(clusterAssment[:,0].A==cent)[0]：找到所有数据中对应clusterAssment中第cent列的行索引
            # dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]就是所有数据的符合条件的具体点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0) #赋值centroid到均值位置
    return centroids, clusterAssment


dataMat=loadDataSet('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch10/testSet.txt')
# init_centroids=randCent(dataMat,4) #4*2的簇
centroids, clusterAssment=kMeans(dataMat,4) #4*2
# print(centroids)

# a=array([[1,0,0],[0,21,0],[3,7,9]])#[0,1,2,2,2],[0,1,0,1,2]:3,7,9都非0，所以第一个array补齐2
# # a = array([[1, 0], [2, 0], [0, 9]])#不为0元素的行索引，放在第一个array[0,1,2]，列索引，放在第二个array[0,0,1]
# b = nonzero(a)
# print(b)


array1=array([[1,2,9],[1,6,7],[2,5,6]])
mat1=mat(array1)
b1=nonzero(mat1[:,0]==1)#nonzero([[True],[True],[False]])=>存在的行索引array(0,1),存在的列索引array(0,0)
row=b1[0]
print(mat1[row])








