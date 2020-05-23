import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# standings=pd.read_csv('leagues_NBA_2013_standings_expanded-standings.csv')
dataset=pd.read_csv('NBA_2014_games.csv',parse_dates=["Date"])
dataset["HomeWin"] = dataset["Visitor/Neutral"] < dataset["Home/Neutral"]
won_last=defaultdict(int)
y_true = dataset["HomeWin"].values

for index,row in dataset.iterrows():
    home_teamName=row['Home/Neutral']
    visit_teamName=row['Visitor/Neutral']
    # 最后一场比赛的胜负情况:多了两个新特征：HomeLastWin和VisitLastWin.
    dataset.loc[index,'HomeLastWin']=won_last[home_teamName]
    dataset.loc[index,'VisitorLastWin'] = won_last[visit_teamName]
    won_last[home_teamName]=row['HomeWin']
    won_last[visit_teamName]=not row['HomeWin']

#5行2列
#print(dataset.iloc[:5,:2])
# array1=np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
# print(array1[:,:2])
# print(array1[:2,:])

#根据上一场比赛的主场和客场胜负情况，判断下一场胜负
clf = DecisionTreeClassifier(random_state=14)
X_previouswins = dataset[["HomeLastWin", "VisitorLastWin"]].values
scores = cross_val_score(clf, X_previouswins, y_true,scoring='accuracy')
# print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))










