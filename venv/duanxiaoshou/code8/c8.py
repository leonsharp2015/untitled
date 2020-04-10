import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from sklearn.svm import SVC

# line=np.linspace(-5,5,200)
# plt.plot(line,np.tanh(line),label='tanh') #tanh()为双曲正切
# plt.plot(line,np.maximum(line,0),label='relu') #x，y逐位比较取最大值
# plt.legend(loc='best')
# plt.show()

wine=load_wine()
X=wine['data'][:,:2]
y=wine['target']
clf=SVC(kernel='rbf',gamma=5,C=10)
clf.fit(X,y)

xx=np.arange(X[:,0].min(),X[:,0].max(),0.1)
yy=np.arange(X[:,1].min(),X[:,1].max(),0.1)
xx,yy=np.meshgrid(xx,yy)
X_grid=np.c_[xx.ravel(),yy.ravel()]
y_grid=clf.predict(X_grid)
plt.gca().contour(xx,yy,y_grid.reshape(xx.shape),alpha=0.8)
plt.pcolormesh(xx,yy,y_grid.reshape(xx.shape),cmap=mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF']))
plt.scatter(X[:,0],X[:,1],c=y,marker='*')
plt.show()











