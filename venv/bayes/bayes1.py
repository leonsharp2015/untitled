#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *
import re
def loadData():
    #6个文档
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocaList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):#所有数据集合，子集合.返回子集合在所有集合是否存在
    returnVec=[0]*len(vocabList) #list:[0,0,......0]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix,traincategory):
    #p(c|w)=p(w|c)*p(c)/p(w)
    #判断文档属于哪个类型，可以把文档内的单词条件概率加和：
    #p(c|(w1,w2))=p((w1,w2)|c)*p(c)/p(w1,w2)=p(w1|c)*p(w2|c)*p(c)/p(w1)*p(w2)
    #log:
    #log(p(c|(w1,w2)))=log(p(w1|c))+log(p(w2|c))+log(p(c))-0-0
    numTrainDocs=len(trainMatrix) # 6
    numWords=len(trainMatrix[0]) #词汇表 32
    pAbusis=sum(traincategory)/float(numTrainDocs) #3,6:侮辱性文档数量／总文档数量 p(c)
    p0Num=zeros(numWords)
    p1Num=zeros(numWords)
    p0Demon=0
    p1Demon=0
    for i in range(numTrainDocs):
        if traincategory[i]==1:
            p1Num+=trainMatrix[i] #p1Num：累加:类别1的doc的单词在词汇表的总计存在个数:[3,0,0....2,1,0,0,0] 32项
            p1Demon+=sum(trainMatrix[i])#累加：类别1的doc内单词的总个数
        else:
            p0Num+=trainMatrix[i]
            p0Demon+=sum(trainMatrix[i])
    p1Vec=p1Num/p1Demon #[p(w1/c1),p(w2/c1),....,p(wn/c1)] 每个单词在类别1中的条件概率
    p0Vec=p0Num/p0Demon
    return p0Vec,p1Vec,pAbusis

def classfiNB(vec2Classify,p0Vec,p1Vec,pClass1):#vec2Classify:验证文档内的单词在词汇表中的存在：[1,0,0,0,1,0,...]
    p1=sum(vec2Classify*p1Vec)+log(pClass1)#sum(1*p(w1|c)+0*p(w2|c)+1*p(w3|c).....)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def bayes_main():
    listOPosts,listClasses=loadData()
    myVocaList=createVocaList(listOPosts)#去除重复元素的所有元素集合:词汇表
    trainMatrix=[]
    for positionDoc in listOPosts:#每个文档
        xx=setOfWords2Vec(myVocaList,positionDoc)
        trainMatrix.append(xx) #trainMatrix:每个doc在词汇表的存在项[0,1,0,0,......0,1]

    p0V,p1V,pAb=trainNB0(trainMatrix,listClasses)
    # testEntry=['love','my','dalmation']
    testEntry=['stupid','garbge']#文档doc
    thisDoc=array(setOfWords2Vec(myVocaList,testEntry))
    c=classfiNB(thisDoc,p0V,p1V,pAb)



def convert_txt():
    mySend='This book is best book on Python or M.L I have ever laid eyes upon.'
    regex=re.compile('\\W*')
    listToken=regex.split(mySend)
    aa=[t.lower() for t in listToken if len(t)>0]
    emailTxt=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch04/email/ham/6.txt').read()
    listToken=regex.split(emailTxt)

def textParse(bigString):
    listOfTokens=re.split(r'\W*',bigString)
    return [token.lower() for token in listOfTokens if len(token)>2]

def spamText():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):#1-25
        #垃圾邮件
        wordList=textParse(open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch04/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        #正常邮件
        wordList=textParse(open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch04/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocaList(docList)#词汇表
    trainSet=list(range(50))#[0,1,2,...,49]
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])#[0-49]10个随机数
        del(trainSet[randIndex])

    trainMat=[]
    trainClass=[]
    for docIndex in trainSet:#[0,1,2,...,49]
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0,p1,pSpam=trainNB0(array(trainMat),array(trainClass))

    for docIndex in testSet:#[0-49]10个随机数
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        c=classfiNB(array(wordVector),p0,p1,pSpam)
        c0=classList[docIndex]


spamText()

















