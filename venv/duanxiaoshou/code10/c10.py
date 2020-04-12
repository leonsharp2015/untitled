import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder

fruits=pd.DataFrame({'数值特征':[5,6,7,8,9],'类型特征':['西瓜','香蕉','桔子','苹果','葡萄']})
fruits_dum=pd.get_dummies(fruits)

rnd=np.random.RandomState(38)
x=rnd.uniform(-5,5,size=50)
y_no_noise=np.cos(6*x)+x
X=x.reshape(-1,1)
y=(y_no_noise+rnd.normal(size=len(x)))/2


clf1=KNeighborsRegressor()
clf1.fit(X,y)
clf2=MLPRegressor()
clf2.fit(X,y)

X_1=np.arange(-5,5,0.1).reshape(-1,1)
y_1=clf1.predict(X_1)
y_2=clf2.predict(X_1)
plt.scatter(x,y,c='r',marker='*')
plt.plot(X_1,y_1,c='y')
plt.plot(X_1,y_2,c='b')

#分箱:X值变为箱的索引的对应:0，1。
bins=np.linspace(-5,5,11)
target_bin=np.digitize(X,bins=bins)
onehot=OneHotEncoder(sparse=False)
onehot.fit(target_bin)
#X值变为x值对应箱的索引,变为的特征矩阵
X_in_bin=onehot.transform(target_bin)
clf3=MLPRegressor()
clf3.fit(X_in_bin,y)

X_3=onehot.transform(np.digitize(np.arange(-5,5,0.1).reshape(-1,1),bins=bins))
y_3=clf3.predict(X_3)
plt.plot(X_1,y_3,c='g')
plt.show()






