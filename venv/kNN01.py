from numpy import *
import operator
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify01(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistance=distances.argsort() #distances的值排序后，输出排序值的索引
    classCount={}
    for i in range(k):
        voteLabel=labels[sortedDistance[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]



g,lbl=createDataSet()
c1=classify01([0,0],g,lbl,3)
v1=[1,2]
# print(tile(v1,(2,2)))
# print(c1)
