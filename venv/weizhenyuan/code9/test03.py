from pandas import read_csv
import numpy as np
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]

# test=SelectKBest(score_func=chi2,k=4)#检验一列对结果class的相关性，卡方值越大， 越不符合；
# fit2=test.fit(X,Y)
# set_printoptions(precision=3)
# print(fit2.scores_) #0-8列每个列的卡方评分，分值越大，数据特征越好:列索引4,1,7,5分数最大
# feathers=fit2.transform(X) #将分值最大的4列打印出来
# print(feathers)

#以逻辑回归算法为基模型，通过递归特征消除来选定对预测结果影响最大的三个数据特征
# model=LogisticRegression()
# rfe=RFE(model,4)
# fit=rfe.fit(X,Y)
# print('特征个数：')
# print(fit.n_features_)
# print('被选定的特征:')
# print(fit.support_)#列索引0,1,5,6是RFE选定的对预测结果影响最大的三个数据特征
# print('特征排名:')
# print(fit.ranking_)

# pca=PCA(n_components=4)
# fit=pca.fit(X)
# X_reduction=pca.transform(X)#降维
# X_2=pca.inverse_transform(X_reduction)#升维
# components=pca.components_  #返回模型特征向量,重新构造出来的维特征
# ratio = pca.explained_variance_ratio_   #返回各个成分各子方差百分比
# a=np.cumsum(ratio)  #累加

#不降维和降维的得分对比
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
model=LogisticRegression(max_iter=3000)
model.fit(X_train,Y_train) #X_train:614*8
score1=model.score(X_test, Y_test) #154*8

pca=PCA(n_components=4) #用4个特征进行train和test
pca.fit(X_train, Y_train)
model=LogisticRegression(max_iter=3000)
X_train_dunction = pca.transform(X_train) #降维X_train_dunction:614*4
X_test_dunction = pca.transform(X_test)
model.fit(X_train_dunction, Y_train)
score2=model.score(X_test_dunction, Y_test) #降维X_test_dunction:154*4
print(score1,score2)










