import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



dataset=pd.read_csv('NBA_2014_games.csv',parse_dates=["Date"])
encoding = LabelEncoder()
encoding.fit(dataset["Home/Neutral"].values)
home_teams = encoding.transform(dataset["Home/Neutral"].values) #能把字符串类型的球队名转化为整型

encoding.fit(dataset["Visitor/Neutral"].values)
visitor_teams = encoding.transform(dataset["Visitor/Neutral"].values)
X_teams = np.vstack([home_teams, visitor_teams]).T #向量组合

onehot = OneHotEncoder()
X_teams_expanded = onehot.fit_transform(X_teams).todense()
print(X_teams_expanded)













