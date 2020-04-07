from pandas import read_csv
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from pickle import dump
from pickle import load
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.metrics import mean_squared_error




filename='housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data=read_csv(filename,names=names,delim_whitespace=True)
array=data.values
X=array[:,0:13]
Y=array[:,13]
set_option('display.width',120)
set_option('precision',2)
# print(data.corr(method='pearson'))
#直方图
# data.hist(sharex=False,sharey=False,xlabelsize=1,ylabelsize=1)
# pyplot.show()
#密度图
# data.plot(kind='density',subplots=True,layout=(4,4),sharex=False,fontsize=1)
# pyplot.show()
#box图
# data.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,fontsize=8)
# pyplot.show()

# scatter_matrix(data)
# pyplot.show()
#相关矩阵
# fig=pyplot.figure()
# ax=fig.add_subplot(111)
# cax=ax.matshow(data.corr(),vmin=-1,vmax=1,interpolation='none')
# fig.colorbar(cax)
# ticks=np.arange(0,14,1)
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names)
# ax.set_yticklabels(names)
# pyplot.show()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=4)
results=[]
# models={}
# models['LR']=LinearRegression()
# models['LASSO']=Lasso()
# models['EN']=ElasticNet()
# models['KNN']=KNeighborsRegressor()
# models['SVM']=SVR()
# for key in models:
#     kfold=KFold(n_splits=10,random_state=7,shuffle=True)
#     cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='neg_mean_squared_error')
#     results.append(cv_result)
#     print('%s:%f (%f)' % (key,cv_result.mean(),cv_result.std()))

# fig=pyplot.figure()
# fig.suptitle('Regression')
# ax=fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(models.keys())
# pyplot.show()

# pipelines={}
# pipelines['ScalerLR']=Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
# pipelines['ScalerLASSO']=Pipeline([('Scaler',StandardScaler()),('LASSO',Lasso())])
# pipelines['ScalerEN']=Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])
# pipelines['ScalerKNN']=Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
# pipelines['ScalerCART']=Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])
# pipelines['ScalerSVM']=Pipeline([('Scaler',StandardScaler()),('SVM',SVR())])
#
# for key in pipelines:
#     kfold=KFold(n_splits=10,random_state=7,shuffle=True)
#     cv_result=cross_val_score(pipelines[key],X_train,Y_train,cv=kfold,scoring='neg_mean_squared_error')
#     results.append(cv_result)
#     print('%s:%f (%f)' % (key,cv_result.mean(),cv_result.std()))
#
# fig=pyplot.figure()
# fig.suptitle('Regression')
# ax=fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(pipelines.keys())
# pyplot.show()

#调参数改变knn,最好为3，-21.616286492374726，14.059167
# scaler=StandardScaler().fit(X_train)
# rescaledX=scaler.transform(X_train)
# para_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
# model=KNeighborsRegressor()
# kfold=KFold(n_splits=10,random_state=7,shuffle=True)
# grid=GridSearchCV(estimator=model,param_grid=para_grid, scoring='neg_mean_squared_error',cv=kfold)
# grid_result=grid.fit(X=rescaledX,y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_,grid_result.best_params_))
# cv_result=zip(grid_result.cv_results_['mean_test_score'],
#               grid_result.cv_results_['std_test_score'],
#               grid_result.cv_results_['params'])
#
# for mean,std,param in cv_result:
#     print('%f (%f) with %r' % (mean,std,param))

#集成算法:GBR -8.825455 (3.633479),RFR -11.320035 (5.937915)
# ensembles={}
# ensembles['ScaledAB']=Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
# ensembles['ScaledAB-KNN']=Pipeline([('Scaler',StandardScaler()),('ABKNN',AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=3)))])
# ensembles['ScaledAB-LR']=Pipeline([('Scaler',StandardScaler()),('ABLR',AdaBoostRegressor(LinearRegression()))])
# ensembles['ScaledRFR']=Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestRegressor())])
# ensembles['ScaledETR']=Pipeline([('Scaler',StandardScaler()),('ETR',ExtraTreesRegressor())])
# ensembles['ScaledGBR']=Pipeline([('Scaler',StandardScaler()),('GBR',GradientBoostingRegressor())])
# for key in ensembles:
#     kfold=KFold(n_splits=10,random_state=7,shuffle=True)
#     cv_result=cross_val_score(ensembles[key],X_train,Y_train,cv=kfold,scoring='neg_mean_squared_error')
#     results.append(cv_result)
#     print('%s:%f (%f)' % (key,cv_result.mean(),cv_result.std()))
#
# fig=pyplot.figure()
# fig.suptitle('Boost')
# ax=fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(ensembles.keys())
# pyplot.show()

#调参，集成算法随机梯度上升GBM:最优：-8.619565823771639 使用{'n_estimators': 600}
# scaler=StandardScaler().fit(X_train)
# rescaledX=scaler.transform(X_train)
# para_grid={'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
# model=GradientBoostingRegressor()
# kfold=KFold(n_splits=10,random_state=7,shuffle=True)
# grid=GridSearchCV(estimator=model,param_grid=para_grid, scoring='neg_mean_squared_error',cv=kfold)
# grid_result=grid.fit(X=rescaledX,y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_,grid_result.best_params_))

#调参，集成算法极端随机树ET:最优：-10.422830918003564 使用{'n_estimators': 10}
# scaler=StandardScaler().fit(X_train)
# rescaledX=scaler.transform(X_train)
# para_grid={'n_estimators':[5,10,20,30,40,50,60,70,80,90,100]}
# model=ExtraTreesRegressor()
# kfold=KFold(n_splits=10,random_state=7,shuffle=True)
# grid=GridSearchCV(estimator=model,param_grid=para_grid, scoring='neg_mean_squared_error',cv=kfold)
# grid_result=grid.fit(X=rescaledX,y=Y_train)
# print('最优：%s 使用%s' % (grid_result.best_score_,grid_result.best_params_))

#最终选定GBM，n_estimators＝600
scaler=StandardScaler().fit(X_train)
rescaledX_train=scaler.transform(X_train)
gbr=GradientBoostingRegressor(n_estimators=600)
gbr.fit(X=rescaledX_train,y=Y_train)

#验证
rescaledX_test=scaler.transform(X_test)
predict=gbr.predict(rescaledX_test)
print(mean_squared_error(Y_test, predict))




































