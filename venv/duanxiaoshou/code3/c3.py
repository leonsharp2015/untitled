from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split

data=make_blobs(n_samples=200,centers=2,random_state=8,cluster_std=[1.0,3.0])
X,y=data #X:200*2包含横纵坐标，y是0，1值
clf=KNeighborsClassifier()
clf.fit(X,y)
#画点
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')
# plt.show()

#网格坐标xx,yy
xx,yy=np.meshgrid(np.arange(X[:,0].min()-1,X[:,0].max()+1,0.02),
                  np.arange(X[:,1].min()-1,X[:,1].max()+1,0.02))  #xx,yy都是(512, 397).xx,yy的每一行都一样。
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]) #ravel将多维数组变为一维.c_[a,b]将a,b连接起来，形成2列横纵坐标,512*397行的网格:(203264, 2).clf.predict([[x],[y]])得到1*203264
Z=Z.reshape(xx.shape)#将1*203264变成(512, 397),作为颜色值

plt.pcolormesh(xx,yy,Z)
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')
plt.xlim(xx.min(),xx.max()) #x轴上的最小值最大值
plt.ylim(yy.min(),yy.max())
plt.show()


# X = [-0.11, -0.06, -0.07, -0.12]
# Y = [0.09, 0.13, 0.17, 0.09]
# Z = [0.1, 0.1, 0.1, 0.1]
# xx, yy = np.meshgrid(X, Y) #xx,yy: 4*4
# print(xx)#4*4,每行相同
# print(xx.ravel())# 从每行连接每行，变成1维
# print(np.c_[xx.ravel(),yy.ravel()]) #将1维的xx,1维的yy组成16*2
# fig, ax = plt.subplots()
# ax.scatter(X, Y, c=Z) #Z颜色
# z_color=np.cos(xx*yy) #4*4
# plt.pcolormesh(X,Y,z_color) #X，Y：指的是二维网格面每一个点的横纵坐标,z_color:(X,Y)坐标处的颜色值
# plt.show()












