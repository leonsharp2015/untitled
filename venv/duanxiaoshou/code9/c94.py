import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

data=make_blobs(n_samples=50,n_features=2,centers=1,random_state=10)
X,y=data
clf=KMeans(n_clusters=3)
clf.fit(X,y)

xx=np.arange(X[:,0].min(),X[:,0].max(),0.2)
yy=np.arange(X[:,1].min(),X[:,1].max(),0.2)
xx,yy=np.meshgrid(xx,yy)
z_grid=np.c_[xx.ravel(),yy.ravel()]
z_predict=clf.predict(z_grid)
plt.pcolormesh(xx,yy,z_predict.reshape(xx.shape))
plt.scatter(X[:,0],X[:,1],c='r')
centroids=clf.cluster_centers_
plt.scatter(centroids[:,0],centroids[:,1],c='w',marker='*')
plt.show()





