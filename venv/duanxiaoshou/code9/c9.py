import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# data=make_blobs(n_samples=50,centers=2,n_features=2,cluster_std=2,random_state=50)
# X,y=data
# fig=plt.figure()
# ax1=fig.add_subplot(1,3,1)
# ax1.set_title('normal')
# ax1.scatter(X[:,0],X[:,1],c=y)
#
# ax2=fig.add_subplot(1,3,2)
# ax2.set_title('Standard')
# X1=StandardScaler().fit_transform(X)
# ax2.scatter(X1[:,0],X1[:,1],c=y)
#
# ax3=fig.add_subplot(1,3,3)
# ax3.set_title('MinMax')
# X1=MinMaxScaler().fit_transform(X)
# ax3.scatter(X1[:,0],X1[:,1],c=y)
# plt.show()

#pca
# wine=load_wine()
# X,y=wine['data'],wine['target']
# X_scaled=StandardScaler().fit_transform(X)
#
# fig=plt.figure()
# ax1=fig.add_subplot(1,2,1)
# pca=PCA(n_components=2)
# pca.fit(X_scaled)
# X_pca=pca.transform(X_scaled)
# #每一类的切片，返回wine的行索引
# X0=X_pca[wine.target==0]
# X1=X_pca[wine.target==1]
# X2=X_pca[wine.target==2]
# ax1.scatter(X0[:,0],X0[:,1],c='b',s=60,edgecolors='k')
# ax1.scatter(X1[:,0],X1[:,1],c='g',s=60,edgecolors='k')
# ax1.scatter(X2[:,0],X2[:,1],c='r',s=60,edgecolors='k')
# ax1.legend(wine.target_names,loc='best')
# ax2=fig.add_subplot(1,2,2)
# ax2.scatter(X_scaled[:,0],X_scaled[:,1],c=y,s=60)
# plt.show()

faces=fetch_lfw_people(min_faces_per_person=20,resize=0.8)
image_shape=faces.images[0].shape
fig,axes=plt.subplots(3,4,figsize=(12,9),subplot_kw={'xticks':(),'yticks':()})
for target,image,ax in zip(faces.target,faces.images,axes.ravel()):
    ax.imshow(image)
ax.set_title(faces.target_names[target])
plt.show()



