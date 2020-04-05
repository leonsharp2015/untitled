from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import set_printoptions
from numpy import mat
import numpy as np
from sklearn.preprocessing import MinMaxScaler

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]
transformer=MinMaxScaler(feature_range=(0,1))
newX=transformer.fit_transform(X)
set_printoptions(precision=3)#小数3位

#list->ndarray->dataframe
# list1=[[1,2,3],[4,5,6],[7,8,10]]
# ndarray1=np.array(list1)
# mat1=mat(ndarray1) #矩阵对象是继承ndarray而来
# df=DataFrame(ndarray1,columns=['c1','c2','c3'])
# print(df)
# df.plot(kind='box',subplots=True,layout=(1,3),sharex=False,sharey=False)
# plt.show()

# df1=DataFrame(newX,columns=names[0:8])
# df1.plot(kind='box',subplots=True,layout=(3,3),sharex=False,sharey=False)
# df1.hist()
# plt.show()


#3组结果的显示
# results=[]
# results.append([1,2,3,4,5])
# results.append([6,7,8,9,10])
# results.append([11,12,13,14,15])
# fig=plt.figure()
# fig.suptitle('array')
# ax=fig.add_subplot(111)#“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
# plt.boxplot(results)
# ax.set_xticklabels(labels=['result1','result2','result3']) #x轴的名称，位置必须boxplot之后
# plt.show()


result_list1=[[1,2,3],[4,5,6],[7,8,10]]
fig=plt.figure()
fig.suptitle('aa')
ax=fig.add_subplot(111)#“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
plt.boxplot(result_list1)
ax.set_xticklabels(labels=['result1','result2','result3']) #x轴的名称，位置必须boxplot之后
plt.show()


























