from numpy import *
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
        ndarray[-1]=int(label)
        index+=1
    return ndarray

ndarray=getClassLabel()
print(ndarray[:,:1])