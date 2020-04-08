from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
data=make_blobs(n_samples=200,centers=2,random_state=8,cluster_std=[1.0,3.0])
X,y=data
# plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')
# plt.show()

clf=KNeighborsClassifier()
clf.fit(X,y)

xx,yy=np.meshgrid(np.arange(X[:,0].min()-1,X[:,0].max()+1,0.02),
                  np.arange(X[:,1].min()-1,X[:,1].max()+1,0.02))
Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]) #ravel将多维数组变为一维,c_[a,b]将a,b连接起来
Z=Z.reshape(xx.shape)
plt.pcolormesh(xx,yy,Z)
plt.scatter(X[:,0],X[:,1],c=y,edgecolors='k')
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.show()















