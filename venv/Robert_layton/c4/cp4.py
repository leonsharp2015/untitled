#!/usr/bin/python
# -*- coding:UTF-8 -*-

import numpy as np
import csv
import os
import sys
import pandas as pd
from collections import defaultdict
data_folder = os.path.expanduser("ml-100k")#http://files.grouplens.org/datasets/movielens/
ratings_filename = os.path.join(data_folder, "u.data")
all_ratings = pd.read_csv(ratings_filename, delimiter="\t",header=None, names = ["UserID", "MovieID", "Rating", "Datetime"])
all_ratings["Datetime"] = pd.to_datetime(all_ratings['Datetime'],unit='s')
all_ratings["Favorable"] = all_ratings["Rating"] > 3
ratings = all_ratings[all_ratings['UserID'].isin(range(200))]
favorable_ratings = ratings[ratings["Favorable"]]#userId在200以内，喜欢的电影：[UserID  MovieID  Rating  Datetime  Favorable]
# print(all_ratings.iloc[:5,:2])

#[userid,{movieid1,moviedid2...}]
favorable_reviews_by_users = dict((uid, frozenset(movies.values))
                                  for uid, movies in favorable_ratings.groupby("UserID")["MovieID"])

#["Favorable"]
num_favorable_by_movie = ratings[["MovieID", "Favorable"]].groupby("MovieID").sum()# num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5]


def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    #电影集合与集合数量
    counts = defaultdict(int)
    # favorable_reviews_by_users:[userid,{movieid1,moviedid2...}]
    # k_1_itemsets:字典:key=1 ,value=[{frozenset([moveid1,moveid2]): 59.0}, frozenset([moveid3,movieid4]): 67.0, frozenset([moveid5,movieid6]): 58.0,....]
    for user, reviews in favorable_reviews_by_users.items():#revirew:某个人喜爱的所有电影列表：{movieid1,moviedid2...}
        for itemset in k_1_itemsets:#itemset:电影同时出现的集合,frozenset([moveid1,moveid2])
            if itemset.issubset(reviews):#itemset中的元素是否都在reviews中
                for other_reviewed_movie in reviews - itemset:
                    # current_superset:frozenset([moveid1, moveid2,movied3])
                    current_superset = itemset | frozenset((other_reviewed_movie,))#current_superset增加元素：frozenset([moveid1,moveid2，。。])
                    counts[current_superset] += 1 #集合itemset出现次数

    #counts字典 key:电影集合([moveid1, moveid2,movied3]),value:电影集合出现次数
    #counts:{frozenset([moveid1, moveid2,movied3]): 6, frozenset([moveid3, moveid4,movied5]): 8}
    list=[(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support]
    return dict(list)

frequent_itemsets = {}
min_support = 50 #Favorable要大于50
#集合
#frequent_itemsets字典:key=1 ,value=[{frozenset([286L]): 59.0, frozenset([7L]): 67.0, frozenset([64L]): 58.0,....]
frequent_itemsets[1] = dict((frozenset((movie_id,)), row["Favorable"])
                            for movie_id, row in num_favorable_by_movie.iterrows()if row["Favorable"] > min_support)
#movie_id也是iterrows（）的序号
#frozenset是固定集合，存储元组[movie_id]。frozenset可以做为dict的键

# for movie_id, row in num_favorable_by_movie.iterrows():
#     if row["Favorable"] > min_support:
#         print movie_id,row["Favorable"]

# userid_moveid=ratings[["UserID","MovieID"]][12:20].groupby("UserID").sum()
# userid_moveid=ratings[["UserID","MovieID"]][12:20].groupby("UserID")["MovieID"]
# for userid,movieid in userid_moveid:
#     print userid,movieid.values

# print userid_moveid.dtypes
#userid_moveid.dtypes #各列名及类型
#userid_moveid.columns #各列名
# userid_moveid_group=ratings[["UserID","MovieID"]][12:20].groupby("UserID")
#print userid_moveid_group.mean().dtypes

# d1={}
# d1[0]=dict((frozenset((mid,)),row["UserID"]) for mid,row in userid_moveid.iterrows())
#
# s={}
# for index,row in userid_moveid.iterrows():
#     s= frozenset((index,))
#     print row["UserID"],row["MovieID"]
# print len(s)
# u1=1
# b = frozenset((u1,))
# print b

for k in range(2, 20):
    # favorable_reviews_by_users:[userid,{movieid1,moviedid2...}]
    # frequent_itemsets:字典:key=range:2,3,4....20 ,value是电影Id集合和Favorable
    #key=2时
    #value=[{frozenset([moveid1,movied2]): 59.0, frozenset([moveid3,movied4]): 67.0, frozenset([moveid2,movied5]): 58.0,....]

    #cur_frequent_itemsets:每个k,决定电影集合中的电影个数，以及集合个数
    # 当k=3,cur_frequent_itemsets:{frozenset([98, 172L, 7]): 78, frozenset([56, 9, 50L]): 66,....}
    cur_frequent_itemsets = find_frequent_itemsets(favorable_reviews_by_users, frequent_itemsets[k - 1],min_support)
    if len(cur_frequent_itemsets) == 0:
        print("Did not find any frequent itemsets of length {}".format(k))
        sys.stdout.flush()
        break
    else:
        print("I found {} frequent itemsets of length {}".format(len(cur_frequent_itemsets), k))
        sys.stdout.flush()
        frequent_itemsets[k] = cur_frequent_itemsets
del frequent_itemsets[1]

#预测：movied8与集合[movied1,movied2，....]一起出现
#candidate_rules:{frozenset([movied1,movied2....]),movied8}
candidate_rules = [] #candidate_rules:{frozenset([64]), 172),(frozenset([64, 98]), 174),(frozenset([64, 98,91]), 121)..}
for itemset_length, itemset_counts in frequent_itemsets.items():
    for itemset in itemset_counts.keys():#itemset:frozenset([movieid1,moviedid2,moviedid3....])
        for conclusion in itemset:#结论conclusion:movieid1
            premise = itemset - set((conclusion,)) #假设premise:frozenset([moviedid2,moviedid3....]
            candidate_rules.append((premise, conclusion))
print("There are {} candidate rules".format(len(candidate_rules)))


correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)

#favorable_reviews_by_users:[userid,{movieid1,moviedid2...}]
for user, reviews in favorable_reviews_by_users.items():#reviews:某一个用户，喜欢的电影集合
    for candidate_rule in candidate_rules:
        premise, conclusion = candidate_rule
                                            #candidate_rule:(frozenset([64, 50L, 98, 174]), 127))
                                            #premise:frozenset([64, 50L, 98, 174]))
                                            #conclusion:127
        if premise.issubset(reviews):
            if conclusion in reviews:
                correct_counts[candidate_rule] += 1#利用某一个用户喜欢的电影集合，判断{[64, 50L, 98, 174]), 127}是否在喜欢的电影集合内
            else:
                incorrect_counts[candidate_rule] += 1

#rule_confidence:电影集合规则和正确率：50L和[56, 258, 172, 181, 7]同时出现的正确率1.0
#{(frozenset([56, 258, 172, 181, 7]), 50L): 1.0, (frozenset([9, 174]), 100L): 0.8888888888888888,....}
rule_confidence = {candidate_rule: correct_counts[candidate_rule] / float(correct_counts[candidate_rule] +
                                                                          incorrect_counts[candidate_rule])
                                                                            for candidate_rule in candidate_rules}

from operator import itemgetter
sorted_confidence = sorted(rule_confidence.items(),key=itemgetter(1), reverse=True)
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print("Rule: If a person recommends {0} they will also  recommend {1}".format(premise, conclusion))
    print(" - Confidence:{0:.3f}".format(rule_confidence[(premise, conclusion)]))
    print("")














