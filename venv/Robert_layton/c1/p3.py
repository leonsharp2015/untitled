from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from numpy import *
import numpy as np
data_set=load_iris()
X=data_set.data
y=data_set.target

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=4)
# values=set(X[:,1])#元素系列 {}
# z=zip(X,y) #将对象中一一对应的元素打包成一个个元组()，然后返回由这些元组组成的列表
# x1=X[:5,:]
# print(y[:5])



#type(ndarray1)都是ndarray类型
ndarray1=array([1,3,4])
ndarray2=array([[1,2,3,6,100],[1,4,3,6,100],[12,23,13,9,200],[-9,-9,-8,-7,300]])


for array,y1 in zip(X_train, y_train):#array,y1 in zip,分成array和数值
    print(array[1])


# for t in zip(X_train, y_train):# t:(array([6.7, 3.1, 4.7, 1.5]), 1)
#     print(t)
