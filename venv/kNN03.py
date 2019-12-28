from numpy import *
import operator
import os
def img2Vector(file):
    returnVector=zeros((1,1024))
    fr=open(file)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,32*i+j]=int(lineStr[j])
    return returnVector

def getNdarray():
    path='/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/trainingDigits/'
    files=os.listdir(path)
    ndarray = zeros([len(files), 1025])
    rowindex=0
    for name in files:
            label=name.split('_')[0]
            vector=img2Vector(path+name)
            ndarray[rowindex,:1024]=vector
            ndarray[rowindex,-1]=label
            rowindex=rowindex+1
    return ndarray

# ndarray=getNdarray()
# print(ndarray[582:780,1024:1025])#3

def classifi0(inX,ndarray,k):
    mat=tile(inX,(ndarray.shape[0],1))
    ndarray2=ndarray[:,:ndarray.shape[1]-1]
    lblVector=ndarray[:,-1]
    diffMat=(mat-ndarray2)**2
    distance=diffMat.sum(axis=1)**0.5
    sortRowIndex=distance.argsort()
    dic={}
    for i in range(k):
        lbl=lblVector[sortRowIndex[i]]
        dic[lbl]=dic.get(lbl,0)+1
    sort_class=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class[0][0]


# inX=array([1,2,3,4])
# ndarray=array([[1,2,3,6,100],[1,4,3,6,100],[12,23,13,9,200],[-9,-9,-8,-7,300]])
# c=classifi0(inX,ndarray,3)
# print(c)

# inX=img2Vector('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/trainingDigits/3_67.txt')
# ndarray=getNdarray()
# c=classifi0(inX,ndarray,3)
# print(c)



def knn_main():
    ndarray = getNdarray()
    test_path='/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/testDigits/'
    filelist=os.listdir(test_path)
    error=0
    for name in filelist:
        vectorTesting = img2Vector(test_path+name)
        class_lbl=name.split('_')[0]
        c_test=classifi0(vectorTesting,ndarray,3)
        error=int(class_lbl)-c_test+error
    return error

# vectorTesting=img2Vector('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch02/digits/testDigits/4_19.txt')
# ndarray=getNdarray()
# c=classifi0(vectorTesting,ndarray,3)
# print(c)

error=knn_main()
print(error)


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

# ndarray=test_vector()
# inX=array([[21,22,23]])
# test_classifi0(inX,ndarray,3)

# d=array([[63,33,3,-27],[11,22,13,6],[9,12,1,6]])
# x=sorted(d,key=operator.itemgetter(2))#按tuple内的分量排序
# d1=array([[63,33,3,-27],[11,22,13,6],[9,12,1,6]])
# print(d1.argsort()) 按值排序的索引

# a1=array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print(a1[:,:-1])#去除最后一列