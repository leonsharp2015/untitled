from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify01(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2 #ndarray
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistance=distances.argsort() #distances的值排序后，输出排序值的索引
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistance[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def file_ndArray():
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/datingTestSet2.txt')
    arrayLines=fr.readlines()
    numLines=len(arrayLines)
    returnMat=zeros((numLines,3))#ndarray
    index=0
    classLabelVector=[]
    for line in arrayLines:
        line=line.strip() #移除字符串头尾指定的字符,默认为空格或换行符,或字符序列
        listFromline=line.split('\t')
        returnMat[index,:]=listFromline[0:3]
        index+=1
        classLabelVector.append(int(listFromline[-1]))
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minValue=dataSet.min(0)
    print(minValue)
    maxValue=dataSet.max(0)
    ranges=maxValue-minValue
    normalSet=zeros(shape(dataSet))
    m=dataSet.shape(0)
    normalDataSet=dataSet-tile(minValue,(m,1))
    normalDataSet=normalDataSet/tile(ranges,(m,1))


# g,lbl=createDataSet()
# c1=classify01([0,0],g,lbl,3)

dataMatrix,datalabels=file_ndArray()
# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(dataMatrix[:,1],dataMatrix[:,2],15.0*array(datalabels),15.0*array(datalabels))
# ax.axis([-2,25,-0.2,2.0])
# plt.show()

ndarray=array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
data=zeros(shape(ndarray))
minValues=ndarray.min(0)
maxValues=ndarray.max(0)
ranges=maxValues-minValues
normalSet=ndarray-tile(minValues,(4,1))
normalSet=normalSet/tile(ranges,(4,1))
# print(ndarray[1,:])

x1=array([[1,2,3],[4,5,6]])
x2=zeros((2,3))
print(x2[1,:])#第2行所有值

ndarray=array([[1,7,32],[4,5,6],[8,2,0]])
minvalues=ndarray.min(0)
aa=ndarray-tile(minvalues,(3,1))
range=ndarray.max(0)-ndarray.min(0)
bb=aa/tile(range,(3,1))
print(bb)



