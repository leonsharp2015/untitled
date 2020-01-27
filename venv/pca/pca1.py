from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def loadDataSet():
    fr=open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch13/testSet.txt')
    strArr=[line.strip().split('\t') for line in fr.readlines()]
    dataArr=[list(map(float,line)) for line in strArr]
    return mat(dataArr)
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #均值为0
    covMat = cov(meanRemoved, rowvar=0) #协方差矩阵
    eigVals,eigVects = linalg.eig(mat(covMat)) #特征值、特征向量
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects #降维，数据的新坐标 transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals #将降维的数据新坐标，转为原坐标
    return lowDDataMat, reconMat

dataMat=loadDataSet() #1000*2
lowDataMat,reconMat=pca(dataMat,1)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')#对比
plt.show()



# data=array([[10.235186,11.321997],
#             [10.122339,11.810993],
#             [9.190236,8.904943],
#             [9.306371,9.847394],
#             [8.330131,8.340352]])
# mat1=mat(data)
# meanVals = mean(mat1, axis=0)#列均值
# meanRemoved = mat1 - meanVals
# covMat = cov(meanRemoved, rowvar=0)  #列的协方差矩阵
# eigVals, eigVects = linalg.eig(covMat)  #协方差矩阵的特征值[0.04696537 2.80402491],特征向量结果eigVects的列,即结果[[-0.89359567 -0.44887279][ 0.44887279 -0.89359567]]的转秩
#
# eigValInd = argsort(eigVals)  # 排序按特征值大小的列索引sort, sort goes smallest to largest
# eigValInd = eigValInd[:-(2 + 1):-1]  # 移除top2以后的特征值 cut off unwanted dimensions
# redEigVects = eigVects[:,eigValInd]  # top2特征值对应的特征向量 reorganize eig vects largest to smallest
# lowDDataMat = meanRemoved * redEigVects  # 原矩阵*列的协方差矩阵的特征向量＝降维的原数据在新空间的坐标   transform data into new dimensions
# reconMat = (lowDDataMat * redEigVects.T) + meanVals


# data=array([[-1,1,0],[-4,3,0],[1,0,2]])
# m2=mat(data)
# m2_mean=mean(m2,axis=0)
# c_m2=m2-m2_mean
# print(sum(c_m2,axis=0))

# data=array([[1,-2],[1,4]])
# mat1=mat(data)
# eigVals,eigVects = linalg.eig(mat1) #特征值,特征向量=eigVects.T
# print(eigVals,eigVects.T)




