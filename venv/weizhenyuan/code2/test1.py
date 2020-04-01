from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

fileName='iris.data.csv'
names=['separ-length','separ-width','petal-length','petal-width','class']
dataset=read_csv(fileName,names=names)
# print(dataset)
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# pyplot.show()
# dataset.hist()
# pyplot.show()
# scatter_matrix(dataset)
# pyplot.show()
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=seed)
models={}
models['LR']=LogisticRegression(max_iter=3000) #算法收敛最大迭代次数
models['LDA']=LinearDiscriminantAnalysis()
models['KNN']=KNeighborsClassifier()
models['CART']=DecisionTreeClassifier()
models['NB']=GaussianNB()
models['SVM']=SVC()
results=[]
for key in models:
    kfold=KFold(n_splits=10,random_state=seed,shuffle=True)
    #每个cv_result包含10个结果
    cv_result=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_result)
    print('%s:%f (%f)'%(key,cv_result.mean(),cv_result.std()))

fig=pyplot.figure()
fig.suptitle('aa')
ax=fig.add_subplot(111)#“111”表示“1×1网格，第一子图”，“234”表示“2×3网格，第四子图”。
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()

# X=np.arange(24).reshape(12,2)#[0-24]变成12行2列
# y=np.random.choice([1,2],12,p=[0.4,0.6]) #1或2，共12个，1出现概率0.4
# y=np.random.choice(np.arange(8),20) #数值范围0-8，20个,正态分布
# kf = KFold(n_splits=5,random_state=7,shuffle=True)#K折交叉验证:将集合切割5份
# for train_index,test_index in kf.split(X):
#     print('train_index:%s,test_index:%s' %(train_index,test_index))

#ConvergenceWarning: lbfgs failed to converge (status=1):STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
#说的是lbfgs 无法收敛，要求增加迭代次数。LogisticRegression里有一个max_iter（最大迭代次数）可以设置，默认为1000。所以在此可以将其设为3000。
#cv_result会得到根据kf分割成5个数据集的5个结果[0.33333333 0. 0. 0. 0. ]
# cv_result=cross_val_score(LogisticRegression(),X,y,cv=kf,scoring='accuracy')
# print(cv_result)











