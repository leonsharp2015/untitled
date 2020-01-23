from numpy import *
import copy

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
    D_copy = copy.deepcopy(D)
    can_list=list(Ck)
    ssCnt = {}
    for tid in D: #[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
        for can in can_list:#can_list:[1],[2],[3],[4],[5]
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = len(list(D_copy))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


ds=loadDataSet()
D=map(set,ds) #python3 map(set,ds)是迭代器.[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
c1=createC1(ds) #[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
#L1:每个单物品项集至少出现在50%以上的记录中
L1,suppd=scanD(D,c1,0.5)#[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
print(L1)


