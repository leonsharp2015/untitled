from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

#测试的回归数据
# X,y=make_regression(n_samples=10,n_features=1,random_state=8,noise=10)

#多变量线性函数：y=w1*x1+w2*x2+b
# X=np.array([[1,12,19,5],[2,24,89,3]])
# y=np.array([4,5])
# lr=LinearRegression()
# lr.fit(X,y)
# X_new=[[8,9,13,7]]
# y_new=lr.predict(X_new)
# print(y_new,lr.coef_[0],lr.coef_[1],lr.coef_[2],lr.coef_[3],lr.intercept_)

#单变量线性函数:直线方程y=wx+b
# X=np.array([[8],[12],[13],[45]])
# y=np.array([0.1,0.3,2,5])
# lr=LinearRegression()
# lr.fit(X,y)
# print(lr.coef_[0],lr.intercept_)#打印w和b
# #对线型函数画线
# x_ticks=np.linspace(8,45,10)
# x_ticks=x_ticks.reshape((-1,1))
# y_ticks=lr.predict(x_ticks)
# plt.plot(x_ticks,y_ticks,c='k')
# plt.scatter(X,y,c='r')
# plt.show()


data_set=load_boston()
X_train,X_test,y_train,y_test=train_test_split(data_set['data'],data_set['target'])
ridge=Ridge()
ridge.fit(X_train,y_train)
score=ridge.score(X_test,y_test)
print(score)





