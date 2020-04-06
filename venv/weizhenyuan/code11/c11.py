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

from sklearn.decomposition import PCA

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]


# num_folds=10
# seed=7
# model=LogisticRegression(max_iter=3000)
# kfold=KFold(n_splits=num_folds,random_state=seed)
# # scoring='neg_log_loss'#对数损失函数
# scoring='roc_auc'#AUC的值越大，诊断准确性越高。
# result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
# print(result)


#分类结果和实际测得值显示在一个混淆矩阵里
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=4)
# model=LogisticRegression()
# model.fit(X_train,Y_train)
# predicted=model.predict(X_test)
# mat1=confusion_matrix(Y_test,predicted)#254
# classes=['0','1']
# df=pd.DataFrame(data=mat1,index=classes,columns=classes)
# print(df)

#精确率（ precision ）、召回率（ recall ）、Fl 值（ Fl-score )和样本数目（ support)
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=4)
# model=LogisticRegression()
# model.fit(X_train,Y_train)
# predicted=model.predict(X_test)
# report=classification_report(Y_test,predicted)
# print(report)

#决定系数R2如果为0.8 ，则表示回归关系可以解释因变量80%的变异。
filename='housing.csv'
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATIO','B','LSTAT','MEDV']
data=read_csv(filename,names=names,delim_whitespace=True)
array=data.values
X=array[:,0:13]
Y=array[:,13]
kfold=KFold(n_splits=10,random_state=7)
model=LinearRegression()
scoring='r2'
result=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
print(result.mean(),result.std())














