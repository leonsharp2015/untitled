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
from sklearn.utils.validation import as_float_array
#
# data_folder = os.path.expanduser("D:\\PythonProject\\test01\\venv\\2019\\")#
# adult_filename = os.path.join(data_folder, "adult.data")
# adult = pd.read_csv(adult_filename, header=None,
#                         names=["Age", "Work-Class", "fnlwgt",
#                         "Education", "Education-Num",
#                         "Marital-Status", "Occupation",
#                         "Relationship", "Race", "Sex",
#                         "Capital-gain", "Capital-loss",
#                         "Hours-per-week", "Native-Country",
#                         "Earnings-Raw"])
# adult.dropna(how='all', inplace=True)
# # print adult.head(10)
# # print adult["Hours-per-week"].describe()
# #print adult["Work-Class"].unique()
#
# X=adult[["Age","Education-Num","Capital-gain","Capital-loss","Hours-per-week"]]
# y = (adult["Earnings-Raw"] == ' >50K').values #目标类别列表
# #1
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# transformer=SelectKBest(score_func=chi2,k=3) #返回k个最佳特征
# xt_chi2=transformer.fit(X,y)
# k2score= transformer.scores_
#
# from scipy.stats import pearsonr
# def multivariate_pearsonr(X, y):
#     scores, pvalues = [],[]
#     for column in range(X.shape[1]):
#         cur_score,cur_p=pearsonr(X[:,column],y)
#         scores.append(abs(cur_score))
#         pvalues.append(cur_p)
#     return (np.array(scores), np.array(pvalues))
#
# transformer = SelectKBest(score_func=multivariate_pearsonr, k=3) #皮尔逊相关系数
# Xt_pearson = transformer.fit_transform(X, y)
#
# #---广告
# data_filename = os.path.join(data_folder, "ad.data")#http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/
# def convert_number(x):
#     try:
#         return float(x)
#     except ValueError:
#         return np.nan
# from collections import defaultdict
# # 不可 converters = defaultdict(convert_number)
# converters={i: convert_number for i in range(1558)}
# converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
# ads = pd.read_csv(data_filename, header=None, converters=converters)
# ads = ads.applymap(lambda x: np.nan if isinstance(x, str) and not x == "ad." else x)
# ads[[0, 1, 2]] = ads[[0, 1, 2]].astype(float)
# ads = ads.astype(float).dropna() #必须写
# X = ads.drop(1558, axis=1).values
# y = ads[1558]
#
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.cross_validation import cross_val_score
# # clf = DecisionTreeClassifier(random_state=14)
# # scores = cross_val_score(clf, X, y, scoring='accuracy')
# #print("The average score is {:.4f}".format(np.mean(scores)))
#
# #这样的特征也被称为主成分:发现特征之间没有相关性、能够描述数据集，这些特征的方差跟整体方差没有多大差距.
# #主成分往往是其他几个特征的复杂组合
# from sklearn.decomposition import PCA
# pca = PCA(n_components=5)
# Xd = pca.fit_transform(X)
# np.set_printoptions(precision=3, suppress=True)
# #PCA会对返回结果根据方差大小进行排序，返回的第一个特征方差最大，第二个特征方差稍小，以此类推。因此，前几个特征往往就能够解释数据集的大部分信
# #print pca.explained_variance_ratio_


X_test=np.array([[0,2],
                [3,5],
                [6,8],
                [9,11],
                [12,14],
                [15,17],
                [18,20],
                [21,23],
                [24,26],
                [27,29]])
y=as_float_array(X_test)
print(y.shape)

