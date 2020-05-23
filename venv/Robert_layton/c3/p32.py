import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

dataset=pd.read_csv('NBA_2014_games.csv',parse_dates=["Date"])
dataset["HomeWin"] = dataset["Visitor/Neutral"] < dataset["Home/Neutral"]
won_last=defaultdict(int)
y_true = dataset["HomeWin"].values

standings=pd.read_csv('leagues_NBA_2013_standings_expanded-standings.csv')

for index, row in dataset.iterrows():
    home_team = row['Home/Neutral']
    visitor_team=row['Visitor/Neutral']

    # 最后一场比赛的胜负情况:多了两个新特征：HomeLastWin和VisitLastWin.
    dataset.loc[index,'HomeLastWin']=won_last[home_team]
    dataset.loc[index,'VisitorLastWin'] = won_last[visitor_team]
    won_last[home_team]=row['HomeWin']
    won_last[visitor_team]=not row['HomeWin']

    #增加一个特征HomeTeamRanksHigher:主场大于客场的排名
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
    home_rank=0
    vistor_rank=0
    if(standings[standings["Team"] == home_team]["Rk"].values.size>0):
        home_rank = standings[standings["Team"] == home_team]["Rk"].values[0]
    if(standings[standings["Team"] == visitor_team]["Rk"].values.size>0):
        visitor_rank = standings[standings["Team"] == visitor_team]["Rk"].values[0]
    dataset.loc[index,"HomeTeamRanksHigher"]=int(home_rank > visitor_rank)

clf = DecisionTreeClassifier(random_state=14)
X_homehigher = dataset[["HomeLastWin", "VisitorLastWin","HomeTeamRanksHigher"]].values
scores = cross_val_score(clf, X_homehigher, y_true,scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))











