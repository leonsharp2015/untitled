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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from pickle import dump
from pickle import load
from pandas import set_option
from pandas.plotting import scatter_matrix



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
models={}
models['LR']=LinearRegression()
models['LASSO']=Lasso()
models['EN']=ElasticNet()
models['KNN']=KNeighborsRegressor()
models['SVM']=SVR()
results=[]
for key in models:
    kfold=KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='neg_mean_squared_error')
    results.append(cv_result)
    print('%s:%f (%f)' % (key,cv_result.mean(),cv_result.std()))




















