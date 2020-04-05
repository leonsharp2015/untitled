import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix

# plt.plot([1,2,3],[4,5,6])
# plt.show()
filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)

set_option('display.width',100) #横向最多显示100个字符
set_option('precision',2)#显示小数点后的位数
# print(data.head(10))
# print(data.describe())
#print(data.groupby('class').size())

# print(data.corr(method='pearson')) #各列的相关系数
# print(data.skew()) #偏度
# data.hist() #直方图
# data.plot(kind='box',subplots=True,layout=(3,3),sharex=False)#蜡烛图
# correlation=data.corr()
# fig=plt.figure()
# ax=fig.add_subplot('111')
# cax=ax.matshow(correlation,vmin=-1,vmax=1) #相关矩阵图
# fig.colorbar(cax)
# ticks=np.arange(0,9,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)

scatter_matrix(data)#各列的相关系数矩阵
plt.show()

































