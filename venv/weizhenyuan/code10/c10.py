from pandas import read_csv
import numpy as np
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

from sklearn.decomposition import PCA

filename='pima_data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
arrays=data.values
X=arrays[:,0:8]
Y=arrays[:,8]
#分离训练数据集和评估数据集(通常会用于算法的执行效率比较低，或者具有大量数据的时候。)
# test_size=0.33
# seed=4
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)
# model=LogisticRegression(max_iter=3000)
# model.fit(X_train,Y_train)
# result=model.score(X_test,Y_test) #通过模型来预测结果，并与己知的结果进行比较， 来评估算法模型的准确度
# print(result)

#K折交叉验证分离:kf分割成10个数据集,然后循环以1个数据集验证，9个训练得到10个分类准确率结果
# num_folds=10
# seed=7
# kfold=KFold(n_splits=num_folds,random_state=seed)
# model=LogisticRegression(max_iter=3000)
# result=cross_val_score(model,X,Y,cv=kfold)
# print(result,result.mean(),result.std())

#通常会用于平衡评估算法、模型训练的速度及数据集的大小。
#弃一交叉验证分离:每个样本单独作为验证集，其余的N-1 个样本作为训练集(程序死机)
# loocv=LeaveOneOut()
# model=LogisticRegression(max_iter=3000)
# result=cross_val_score(model,X,Y,cv=loocv)
# print(result)

#重复随机分离:按3-7的比例，随机分成10组,计算10组的准确率
n_splits=10
test_size=0.3
seed=7
kfold=ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=seed)
# for train_index,test_index in kfold.split(X):
#     print('train_index:%s,test_index:%s' %(train_index,test_index))
model=LogisticRegression()
result=cross_val_score(model,X,Y,cv=kfold)#10个result结果
print(result)





