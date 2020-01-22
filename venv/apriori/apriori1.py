from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    # 将C1冰冻，即固定列表C1，使其不可变
    return map(frozenset, C1)  # use frozen set so we can use it as a key in a dict

def scanD(D, Ck, minSupport):
    numItems=len(list(D))
    print(numItems)
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                #if not ssCnt.has_key(can): ssCnt[can]=1
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = len(list(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


ds=loadDataSet()
c1=createC1(ds)
D=map(set,ds) #list(map)可以显示map的数据
L1,suppd=scanD(D,c1,1)
print(L1)


