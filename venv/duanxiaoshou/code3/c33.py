from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.datasets import load_wine

wine_set=load_wine() #dict_keys(['feature_names', 'data', 'target_names', 'target', 'DESCR'])
X_train,X_test,y_train,y_test=train_test_split(wine_set['data'],wine_set['target'],random_state=0)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
score=knn.score(X_test,y_test)
X_new=np.array([[13.2,2.77,2.51,18.5,96.6,1.04,2.55,0.57,1.47,6.2,1.05,3.33,820]])
c=knn.predict(X_new)
print(c)

