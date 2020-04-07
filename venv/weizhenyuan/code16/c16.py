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

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]
kfold=KFold(n_splits=10,random_state=7)
#装袋决策
# cart=DecisionTreeClassifier()
# model=BaggingClassifier(base_estimator=cart,n_estimators=100,random_state=7)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result.mean())

#随机森林
# model=RandomForestClassifier(n_estimators=100,random_state=7,max_features=3)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result)

#提升
# model=AdaBoostClassifier(n_estimators=30,random_state=7)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result)

#随机梯度提升GBM
# model=GradientBoostingClassifier(n_estimators=100,random_state=7)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result)

#投票
cart=DecisionTreeClassifier()
models=[]
models.append(('logistic',LogisticRegression()))
models.append(('cart',DecisionTreeClassifier()))
models.append(('svm',SVC()))
ensure_model=VotingClassifier(estimators=models)
result=cross_val_score(ensure_model,X,Y,cv=kfold)
print(result)










