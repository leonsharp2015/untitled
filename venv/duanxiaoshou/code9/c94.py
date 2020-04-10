import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=make_blobs(n_samples=50,n_features=2,centers=1,random_state=1)
X,y=data
kmean=KMeans(n_clusters=3)
kmean.fit(X,y)

xx=np.arange(X[:,0].min(),X[:,0].max(),0.2)
yy=np.arange(X[:,1].min(),X[:,1].max(),0.2)
xx,yy=np.meshgrid(xx,yy)
z_grid=np.c_[xx.ravel(),yy.ravel()]
z_predict=kmean.predict(z_grid)
plt.pcolormesh(xx,yy,z_predict.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c='r')
plt.show()





