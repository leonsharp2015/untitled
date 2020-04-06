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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]
kfold=KFold(n_splits=10,random_state=7)

#Logistic
# model=LogisticRegression()
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())
#线性判别分析
# model=LinearDiscriminantAnalysis()
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())
#k-means
# model=KNeighborsClassifier()
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())
#bayes
# model=GaussianNB()
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())
#decisionTree
# model=DecisionTreeClassifier()
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())
#SVC
model=SVC()
result=cross_val_score(model,X,Y,cv=kfold)
print(result.mean())
























