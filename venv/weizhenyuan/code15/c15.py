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
#数据正态化处理及线性判别的管道
# steps=[]
# steps.append(('Standardize',StandardScaler()))
# steps.append(('lda',LinearDiscriminantAnalysis()))
# model=Pipeline(steps)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result)

#特征选择和生成模型的Pipeline
# features=[]
# features.append(('pca',PCA()))
# features.append(('select_best',SelectKBest(k=6)))
# steps=[]
# steps.append(('feature_union',FeatureUnion(features)))
# steps.append(('logistic',LogisticRegression()))
# model=Pipeline(steps)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result,result.mean())












