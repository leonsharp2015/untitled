from numpy import *
import operator
def file2NdArray():
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/datingTestSet2.txt')
    arrayLine=fr.readlines()
    numLines = len(arrayLine)
    returnMat=zeros((numLines,3))
    lblVector=[]
    index=0
    for line in arrayLine:
        txt=line.strip()
        ndarray=txt.split('\t')
        returnMat[index,:]=ndarray[0:3]
        lblVector.append(ndarray[-1])
        index+=1
    return returnMat,lblVector
def autoNormal(ndarray):
    minValues=ndarray.min(0)
    range=ndarray.max(0)
    ndarray=ndarray-tile(minValues,(ndarray.shape[0],1))
    ndarray=ndarray/tile(range,(ndarray.shape[0],1))
    return ndarray

def classify0(inX,dataSet,lblVector,k):
    mat=tile(inX,(dataSet.shape[0],1))
    diffArray=(mat-dataSet)**2
    diff=diffArray.sum(1)**0.5 #sum(1)对每一行，将所有列加在一起
    sortedDistance=diff.argsort()
    classCount = {}#dictinanry
    for i in range(k):
        c=lblVector[sortedDistance[i]]
        classCount[c]=classCount.get(c,0)+1#dictinary如果没有,返回0
    sortClass=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#按第二个值，也就是value排序
    return sortClass[0][0]#返回dictinary第一个key-value的key

def ndarry_test():
    radio=0.1
    ndarray,lblVector=file2NdArray()
    ndarray=autoNormal(ndarray)
    recordCount=ndarray.shape[0]#1000
    numTestVector=int(recordCount*radio)#100
    for i in range(numTestVector):
        # 测试v1:0到100
        v1=ndarray[i,:]
        lbl=lblVector[i]
        # ndarray是行100-1000，lblVector是行100-1000
        #检测v1在行100-1000的分类结果
        result_c=classify0(v1,ndarray[numTestVector:recordCount,:],lblVector[numTestVector:recordCount],3)
        print(lbl,result_c)



ndarry_test()

# ndarray=array([[20,50,40],[79,19,99],[2,3,1],[39,97,123]])
# lbl=['C','B','C','D']
# inX=[1,5,6]
# c=classify0(inX,ndarray,lbl,3)
# print(c)

# students=[('tony',23,12),('jerrt',9,91),('zeter',1,5)]
# list1=sorted(students,key=operator.itemgetter(0),reverse=True)
# print(list1)

# dic1={}
# dic1["x"]=12
# dic1["y"]=23
# dic1["z"]=18
# sortDic=sorted(dic1.items(),key=operator.itemgetter(1),reverse=True)
# print(sortDic[0][0])





