from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np

# X=np.array([
#     [0,1,0,1],
#     [1,1,1,0],
#     [0,1,1,0],
#     [0,0,0,1],
#     [0,1,1,0],
#     [0,1,0,1],
#     [1,0,0,1]
# ])
# y=np.array([0,1,1,0,1,0,0])
# clf=BernoulliNB() #数据集中的每个特征都只有0和1两个数值，在这种情况下，贝努利贝叶斯的表现还不错
# clf.fit(X,y)
# X_new=np.array([
#     [0,1,1,1],
#     [1,1,0,1]
# ])
# y_new=clf.predict(X_new)

#模拟分类集
# X,y=make_blobs(n_samples=500,centers=5,n_features=2,random_state=8)
# X_train,X_test,y_train,y_test=train_test_split(X,y)
# clf=BernoulliNB()
# clf.fit(X_train,y_train)
# plt.scatter(X[:,0],X[:,1],y,edgecolors='r')
# plt.show()

#网格线
# X=np.array([1,2,3])
# Y=np.array([0.7,0.8,0.9])
# X,Y=np.meshgrid(X,Y)#生成网格坐标
# plt.pcolormesh(X,Y,np.cos(X*Y))#np.cos(X*Y) 网格坐标的颜色
# plt.show()

#观察算法对整个平面坐标的划分
# X,y=make_blobs(n_samples=500,centers=5,n_features=2,cluster_std=[0.2,0.7,0.1,0.6,0.5],random_state=8) #random_state设置可以保证每次执行取得的数据一样
# clf=GaussianNB() #BernoulliNB()
# clf.fit(X,y)
# xx=np.linspace(X[:,0].min(),X[:,0].max(),num=100) # xx=np.arange(X[:,0].min(),X[:,0].max(),0.1)
# yy=np.linspace(X[:,1].min(),X[:,1].max(),num=100)
#
# #用GaussianNB计算对网格坐标的预测值，作为color,可以看出模型对整个平面的划分
# #对比可以用pcolormesh背景（输入网格坐标）或者scatter画网格的点（输入2列网格坐标x,y的值:[[x],[y]]）
# xx,yy=np.meshgrid(xx,yy)
# grid_test=np.c_[xx.reshape(-1,1),yy.reshape(-1,1)]
# z_color=clf.predict(grid_test)
# plt.scatter(grid_test[:,0],grid_test[:,1],z_color)
# # plt.pcolormesh(xx,yy,z_color.reshape(xx.shape))
# plt.scatter(X[:,0],X[:,1],y,c='r',edgecolors='k')
# plt.xlim(X[:,0].min(),X[:,0].max())
# plt.ylim(X[:,1].min(),X[:,1].max())
# plt.show()

#学习曲线
cancer_data=load_breast_cancer()
X,y=cancer_data['data'],cancer_data['target'] #(569, 30)
cv=ShuffleSplit(n_splits=100,test_size=0.2,random_state=0) #n_splits:int, 划分训练集、测试集的次数,test_size表示测试集占总样本的比例
estiator=GaussianNB()
train_size,train_score,test_core=learning_curve(estiator,X,y,cv=cv,n_jobs=4,train_sizes=np.linspace(0.1,1,5))#train_sizes 训练样本的相对的或绝对的数字
train_score_mean=np.mean(train_score,axis=1)
test_score_mean=np.mean(test_core,axis=1)
plt.figure()
plt.ylim(0.9,1.01)
plt.grid()
plt.plot(train_size,train_score_mean,'o-',color='r',label='Training score')
plt.plot(train_size,test_score_mean,'o-',color='g',label='cross-validate score')
plt.legend(loc='lower right')
plt.show()





























