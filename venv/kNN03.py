from numpy import *
import operator
import os
def img2Vector(file):
    returnVector=zeros((1,1024))
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/trainingDigits/'+file)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector

def getClassLabel():
    path='/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/trainingDigits/'
    files=os.listdir(path)
    ndarray = zeros([len(files), 1025])
    index=0
    for name in files:
        label=name.split('_')[0]
        vector=img2Vector(name)
        ndarray[index,:1024]=vector
        ndarray[index,-1]=int(label)
        index+=1
    return ndarray


def test_vertor_one(index):
    v=zeros([1,3])
    for i in range(3):
        v[0,i]=index*10+i
    return v

def test_vector():
    #a1 =ndarray(shape=(2, 2), dtype=float, order='F')#构造类ndarray
    a2=array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    ndarray=zeros([4,4])
    for i in range(a2.shape[0]):
        # ndarray[i, :3] = a2[i, :]
        ndarray[i,:3]=test_vertor_one(i)
        ndarray[i,-1]=i+100#行变化，最后一列
    return ndarray

# print(aa[1:3,0:4])#1-3行，0-4列
# print(aa[:,0:4])#所有行，0-4列

def test_classifi0(inX,ndarray,k):
    mat=tile(inX,(ndarray.shape[0],1))
    diffMat=mat-ndarray[:,:ndarray.shape[1]-1]
    distanceMat=diffMat**2
    distance=distanceMat.sum(axis=1)
    sortIndex=distance.argsort()
    lblVector=ndarray[:,-1]
    dic={}
    for i in range(k):
        lbl=lblVector[sortIndex[i]]
        dic[lbl]=dic.get(lbl,0)+1
    sortClass=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
    print(sortClass[0][1])

ndarray=test_vector()
inX=array([[21,22,23]])
test_classifi0(inX,ndarray,3)

# d=array([[63,33,3,-27],[11,22,13,6],[9,12,1,6]])
# x=sorted(d,key=operator.itemgetter(2))#按tuple内的分量排序
# d1=array([[63,33,3,-27],[11,22,13,6],[9,12,1,6]])
# print(d1.argsort()) 按值排序的索引
