from numpy import *
from math import log
def test():
    ds=array([[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']])
    lbl=['no surfacing','flippers']
    length=ds.shape[0]
    dic={}
    for i in range(len(ds)):
        lbl=ds[i,-1]#每一行，最后一列
        dic[lbl]=dic.get(lbl,0)+1
    shannon=0
    for key in dic.keys():
        p1=(dic.get(key)/length)
        shannon-=p1*log(p1,2)
    print(shannon)

def splitDataSet1():
    ds = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    result=[]
    axis=0
    value=1
    for vec in ds:
        if int(vec[axis]) == value:
            print(vec[:axis])
            # reducedFatVec=vec[:axis]
            # reducedFatVec.extends(vec[axis+1:])
            # result.append(reducedFatVec)

# p2 = [1, 2, 3, 4, 5, 6, 7]
# print(p2[:3])#到第2个元素
# print(p2[3:])#从第3个元素
# retDataSet=[]
# for list in dataSet:
#     # print(list[0:3])#list[0:3]表示list中的第0个元素到第2个元素,也可list[:3]
#     list1=list[:1]
#     list2=list[2:]
#     list1.extend(list2)#会把list2的元素加入集合
#     retDataSet.append(list1)#添加元素
#     print(list1)
#     print('**')
# print(retDataSet)


def splitDataSet(dataSet, axis, value):#每一行去除axis列，条件是axis列值＝value  ＃chop out axis used for splitting
    retDataSet = []
    for featVec in dataSet:#featVec是list类型
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]   #  featVec从0到axis-1的元素
            reducedFeatVec.extend(featVec[axis+1:])#extend将featVec里的元素加入reducedFeatVec
            retDataSet.append(reducedFeatVec)#append将featVec这个list加入
    return retDataSet


# dataSet = [[1, 1, 'yes'],
#            [1, 1, 'yes'],
#            [1, 0, 'no'],
#            [0, 1, 'no'],
#            [0, 1, 'no']]
# aa=splitDataSet(dataSet,0,1)#当第0列＝1，其他的数据是[[1, 'yes'], [1, 'yes'], [0, 'no']]
# print(aa)

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

def chooseBestFeatherToSplit(dataSet):#最大的不确定,该列包含最大的墒
    numFeathers=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInGain=0.0
    bestFeather=-1
    for i in range(numFeathers):
        featList=[row[i] for row in dataSet]#每一行的对应第i列值
        uniqueValues=set(featList)
        newEntropy=0.0
        for value in uniqueValues:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInGain):
            bestInGain=infoGain
            bestFeather=i
    return bestFeather

dataSet = [[1, 1, 'yes'],
           [1, 1, 'yes'],
           [1, 0, 'no'],
           [0, 1, 'no'],
           [0, 1, 'no']]
# aa=chooseBestFeatherToSplit(dataSet)
# d1=[[1,2,3,4],[5,6,7,8]]
# for i in range(len(dataSet[0])-1):
#     featList = [example[i] for example in dataSet]
#     uniqueValues = set(featList)
#     print(featList)
#     print(uniqueValues)
#     print("**")

def test_splitDataSet(ds,axis,value):
    result=[]
    for v in ds:
        if(v[axis]==value):
            list=[]
            beforeV=v[:axis]
            after=v[axis+1:]
            list.extend(beforeV)
            list.extend(after)
            result.append(list)
    return result

def test_calaShannonEnd(dataSet):
    m=len(dataSet)
    lblCount={}
    for v in dataSet:
        lbl=v[-1]
        if lbl not in lblCount.keys():lblCount[lbl]=0
        lblCount[lbl]+=1
    shannon=0
    for lbl in lblCount.keys():
        p1=int(lblCount.get(lbl))/m
        shannon-=p1*log2(p1)
    return shannon

def test_chooseBestLabel(dataSet):
    gain=0
    index=-1
    for vector in dataSet:
        for labelAxis in range(len(vector)):
            value=vector[labelAxis]
            split_ds=test_splitDataSet(dataSet,labelAxis,value)
            g1=test_calaShannonEnd(split_ds)
            print(g1)
            if gain>g1:
                gain=g1
                index=labelAxis


(dataSet)








