import datetime
from numpy import *
from numpy.linalg import matrix_rank
import operator


t=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')
m1=mat(random.rand(4,4))
m2=mat(random.rand(4,4))
m1_inv=m1.I
cc=m1*m1_inv-eye(4)

a = arange(8)
b = a.reshape(2,4)#1-8两行四列ndarray

list=array([
    [1,2,3],[4,5,6],[7,8,10] #ndarray
])
rndlist=random.rand(3,3)#ndarray
m_list=mat(list)#matrix
list2=[[4,2],[3,2],[3,1]]#list
m_list2=mat(list2)

list3=[[1,2,3],[4,5,6],[7,8,10]]
m_list3=mat(list3)
rank=matrix_rank(m_list)
inv_mlist3=m_list3.I
list4=[1,5]
t1=tile(list4,(4,1))

x1=[1,2]
x=[[0,9],[1,3],[7,7],[9,12]]
x_m=mat(x)
x1_m=tile(x1,(4,1))
diff_m=(x1_m-x_m)
print(diff_m**2)









