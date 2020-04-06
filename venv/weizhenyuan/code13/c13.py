from pandas import read_csv
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#线性回归和非线性回归
filename='housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data=read_csv(filename,names=names,delim_whitespace=True)
array=data.values
X=array[:,0:13]
Y=array[:,13]
kfold=KFold(n_splits=10,random_state=7)
# model=LinearRegression() #线性回归
# model=Ridge() #岭回归
# model=Lasso() #lasso
# model=ElasticNet() #弹性网络回归
# model=KNeighborsRegressor() #k-mean回归
# model=DecisionTreeRegressor() #回归树
model=SVR() #支持向量积
scoring='neg_mean_squared_error'
result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print(result.mean())













