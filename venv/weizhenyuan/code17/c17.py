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

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]
kfold=KFold(n_splits=10,random_state=7)

#网格搜索最佳参数
# model=Ridge()
# param_grid={'alpha':[1,0.1,0.01,0.001,0]}
# grid=GridSearchCV(estimator=model,param_grid=param_grid)
# grid.fit(X,Y)
# print(grid.best_score_) #最高得分
# print(grid.best_estimator_.alpha)#最优参数

#随机搜索最佳参数
model=Ridge()
param_grid={'alpha':uniform()}
grid=RandomizedSearchCV(estimator=model,param_distributions=param_grid,n_iter=100,random_state=7)
grid.fit(X,Y)
print(grid.best_score_) #最高得分
print(grid.best_estimator_.alpha)#最优参数






















