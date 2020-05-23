#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import csv
import os
import sys
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


adult = pd.read_csv("adult.data", header=None,
                        names=["Age", "Work-Class", "fnlwgt",
                        "Education", "Education-Num",
                        "Marital-Status", "Occupation",
                        "Relationship", "Race", "Sex",
                        "Capital-gain", "Capital-loss",
                        "Hours-per-week", "Native-Country",
                        "Earnings-Raw"])
adult.dropna(how='all', inplace=True)
#adult["Hours-per-week"].describe() 对某一列求均值、方差等
# X = np.arange(30).reshape((10, 3))
# X[:,1]=1#把所有第二列的数值都改为1
#
# from sklearn.feature_selection import VarianceThreshold
# vt = VarianceThreshold()
# Xt = vt.fit_transform(X)
#print(vt.variances_) #每一列方差

X = adult[["Age", "Education-Num", "Capital-gain", "Capital-loss","Hours-per-week"]].values
y = (adult["Earnings-Raw"] == ' >50K').values #目标类别列表
#表现好的单个特征（单变量），依据是它们各自所能达到的精确度。只要测量变量和目标之间的某种相关性就行。
transformer = SelectKBest(score_func=chi2, k=3) #卡方检验,返回3个最佳特征
Xt_chi2 = transformer.fit_transform(X, y)

#k=3 [8.60061182e+03 2.40142178e+03 8.21924671e+07 1.37214589e+06 6.47640900e+03]
#SelectKBest返回k个最佳特征，相关性最好的分别是第一、三、四列()
# Age（年龄）、Capital-Gain（资本收益）和Capital-Loss（资本损失）.三个特征从单变量特征选取角度来说，这些就是最佳特征
#print(transformer.scores_)

from scipy.stats import pearsonr
def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for column in range(X.shape[1]):
        cur_score, cur_p = pearsonr(X[:, column], y)
        scores.append(abs(cur_score))
        pvalues.append(cur_p)
    return (np.array(scores), np.array(pvalues))

transformer = SelectKBest(score_func=multivariate_pearsonr, k=3) #皮尔逊相关系数,返回3个最佳特征
Xt_pearson = transformer.fit_transform(X, y)
# print(transformer.scores_)

#对比
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=14)
scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring='accuracy')
scores_pearson = cross_val_score(clf, Xt_pearson, y,scoring='accuracy')

#---广告
data_filename = "ad.data"#http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/
def convert_number(x):
    try:
        return float(x)
    except ValueError:
        return np.nan
from collections import defaultdict
# 不可 converters = defaultdict(convert_number)
converters={i: convert_number for i in range(1558)}
converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
ads = pd.read_csv(data_filename, header=None, converters=converters)
ads = ads.applymap(lambda x: np.nan if isinstance(x, str) and not x == "ad." else x)
ads[[0, 1, 2]] = ads[[0, 1, 2]].astype(float)
ads = ads.astype(float).dropna() #必须写
X = ads.drop(1558, axis=1).values
y = ads[1558]


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X, y, scoring='accuracy')
#print("The average score is {:.4f}".format(np.mean(scores)))

#这样的特征也被称为主成分:发现特征之间没有相关性、能够描述数据集，这些特征的方差跟整体方差没有多大差距.
#主成分往往是其他几个特征的复杂组合
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)
np.set_printoptions(precision=3, suppress=True)
#PCA会对返回结果根据方差大小进行排序，返回的第一个特征方差最大，第二个特征方差稍小，以此类推。因此，前几个特征往往就能够解释数据集的大部分信
#print pca.explained_variance_ratio_

clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
print("The average score from the reduced dataset is {:.4f}".format(np.mean(scores_reduced)))

