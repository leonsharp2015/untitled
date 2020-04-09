from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X=np.array([
    [0,1,0,1],
    [1,1,1,0],
    [0,1,1,0],
    [0,0,0,1],
    [0,1,1,0],
    [0,1,0,1],
    [1,0,0,1]
])
y=np.array([0,1,1,0,1,0,0])
clf=BernoulliNB() #数据集中的每个特征都只有0和1两个数值，在这种情况下，贝努利贝叶斯的表现还不错
clf.fit(X,y)
X_new=np.array([
    [0,1,1,1],
    [1,1,0,1]
])
y_new=clf.predict(X_new)

X,y=make_blobs(n_samples=500,centers=5,n_features=2,random_state=8)
X_train,X_test,y_train,y_test=train_test_split(X,y)
clf=BernoulliNB()
clf.fit(X_train,y_train)
score=clf.score(X_test,y_test)
print(score)












