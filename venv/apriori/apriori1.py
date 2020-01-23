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

def scanD(D, Ck, minSupport):#D, Ck都是迭代器
    D_copy = copy.deepcopy(D)
    can_list=list(Ck)
    ssCnt = {}
    for tid in list(D):
        for can in can_list:
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

#------------------------
def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)#遍历器
    D2=map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D2, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData




# ds=loadDataSet()
# D=map(set,ds) #python3 map(set,ds)是迭代器.[[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
# c1=createC1(ds) #[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
# #L1:每个单物品项集至少出现在50%以上的记录中
# L1,suppd=scanD(D,c1,0.75)#[frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
# print(L1)

ds=loadDataSet()
L,suppData=apriori(ds)
print(L[2])


