from numpy import *
from math import log
import operator
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

def chooseBestFeatherToSplit_1(dataSet):#最大的不确定,该列包含最大的墒
    numFeathers=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInGain=0.0
    bestFeather=-1
    for i in range(numFeathers):
        featList=[row[i] for row in dataSet]#每一行的对应第i列值
        uniqueValues=set(featList)
        newEntropy=0.0#每列的墒
        for value in uniqueValues:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
            print(subDataSet,newEntropy)
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
# aa=chooseBestFeatherToSplit_1(dataSet)
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
    numFeathers = len(dataSet[0]) - 1
    baseEntropy=test_calaShannonEnd(dataSet)
    bestInfoGain = 0.0
    bestFeature=-1
    for i in range(numFeathers):
        featherList=[v[i] for v in dataSet ]
        unique_values=set(featherList)
        one_feather_gain=0#一列的墒
        for value in unique_values:
            split_ds=test_splitDataSet(dataSet,i,value)
            one_value_gain=test_calaShannonEnd(split_ds)#某一个值的墒
            prop=len(split_ds)/float(len(dataSet))
            one_feather_gain+=prop*one_value_gain
            # print(split_ds,one_feather_gain)
        infoGain = baseEntropy - one_feather_gain
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    # print('bestFeature',bestFeature)
    return bestFeature

# test_chooseBestLabel(dataSet)
# [[1, 'no'], [1, 'no']] 0.0
# [[1, 'yes'], [1, 'yes'], [0, 'no']] 0.5509775004326937
# 0 0.5509775004326937
# [[1, 'no']] 0.0
# [[1, 'yes'], [1, 'yes'], [0, 'no'], [0, 'no']] 0.8
# 1 0.8

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
            # print(subDataSet,newEntropy)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    # print('bestFeature',bestFeature)
    return bestFeature  # returns an integer

def test_majorityClass(classList):
    dic={}
    for c in classList:
        if c not in dic.keys():dic[c]=0
        dic[c]+=1
    s=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
    print('test_majorityClass:',s[0][0])
    return s[0][0]

def splitDataSet_0(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def majorityCnt_0(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def test_createTree(dadaSet,labels):
    # classList=[v[-1] for v in labels]
    # if classList.count(classList[0])==len(classList):
    #     return classList[0]
    # if len(dataSet[0])==1:
    #     return test_majorityClass(classList)
    # bestFeat=test_chooseBestLabel(dataSet)
    # bestFeatlabel=labels[bestFeat]
    # myTree={bestFeatlabel:{}}
    # del(labels[bestFeat])
    # featureValue=[example[bestFeat] for example in dataSet]
    # unqueValue=set(featureValue)
    # for value in unqueValue:
    #     subLabels=labels[:]
    #     print(subLabels)
    #     myTree[bestFeatlabel][value]=test_createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    # return myTree

    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt_0(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = test_createTree(splitDataSet_0(dataSet, bestFeat, value), subLabels)
    return myTree

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels

ds,labels=createDataSet()
myTree=test_createTree(ds,labels)















