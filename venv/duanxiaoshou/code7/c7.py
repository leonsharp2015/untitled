import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.datasets import load_wine
import matplotlib as mpl


# X,y=make_blobs(n_samples=50,centers=2,random_state=8)
# clf=svm.SVC(kernel='rbf',C=1000)
# clf.fit(X,y)
# plt.scatter(X[:,0],X[:,1],y)
#
# ax=plt.gca()
# xlim=ax.get_xlim()
# ylim=ax.get_ylim()
#
# xx=np.linspace(xlim[0],xlim[1],30)
# yy=np.linspace(ylim[0],ylim[1],30)
# XX,YY=np.meshgrid(xx,yy)
# xy=np.vstack([XX.ravel(),YY.ravel()]).T #vstack [[x],[y]]将两个列叠加成一列
# Z=clf.decision_function(xy).reshape(XX.shape) #decision_function计算样本点到分割超平面的函数距离
#
# ax.contour(XX,YY,Z,color='k',level=[-1,0,1],alpha=0.5,linestyles=['-.','-',':']) #使用XX和YY绘制Z的等高线图
# ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,linewidths=1,facecolors='none')
# plt.show()

wine=load_wine()
X=wine['data'][:,:2]
y=wine['target']
C=1.0
models=(svm.SVC(kernel='linear',C=C),
        svm.LinearSVC(C=C),
        svm.SVC(kernel='rbf',gamma=0.7,C=C),
        svm.SVC(kernel='poly',degree=3,C=C))
models=(clf.fit(X,y) for clf in models)
titles=('svc with linear kernal',
        'linearSVC (linear kernel)',
        'svc with rbf kernel',
        'svc with polynomialkernal')

fig,sub=plt.subplots(2,2)
plt.subplots_adjust(wspace=0.4,hspace=0.4)
xx,yy=X[:,0],X[:,1]
xx=np.arange(xx.min(),xx.max(),0.02)
yy=np.arange(yy.min(),yy.max(),0.02)
xx,yy=np.meshgrid(xx,yy) #(253, 191),(253, 191)

# plt.scatter(xx,yy,np.cos(xx*yy))#xx*yy表示每个元素相乘
# plt.pcolormesh(xx,yy,np.cos(xx*yy))

#指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

clf=svm.SVC(kernel='poly',degree=3,C=C)
clf.fit(X,y)
Z=clf.predict(np.c_[xx.reshape(-1,1),yy.reshape(-1,1)])
plt.pcolormesh(xx,yy,Z.reshape(xx.shape),cmap=cm_light)#背景,位置放在前面绘制 #plt.scatter(xx,yy,Z)???
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap=cm_dark,marker='o',edgecolors='k') #edgecolors是指描绘点的边缘色彩，s指描绘点的大小，cmap指点的颜色
plt.show()



















