from numpy import  *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

def cosSim(inA,inB):#余弦距离：a.b/|a||b|
    num = float(inA.T*inB)#矢量内积 a.b
    denom = la.norm(inA)*la.norm(inB)#|a|*|b|矢量范数
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):#item:user评分为0的列索引
    n = shape(dataMat)[1]
    simTotal = 0.0;
    ratSimTotal = 0.0
    for j in range(n):#dataMat的每一列
        userRating = dataMat[user, j]#该列被user打分
        if userRating == 0:
            continue
        #item 1,2
        #item=1:j=0 overLap=[0,3,4,5,6]  j=3 overLap=[0,3] j=4 overLap=[0]
        overLap = nonzero(logical_and(dataMat[:, item].A > 0,dataMat[:, j].A > 0))[0]#在dataMat所有行中，item列不为0，j列也不为0的行集合。(选出item列有值，j列也同时有值的行集合）
        if len(overLap) == 0:
            similarity = 0
        else:
            v1=dataMat[overLap, item] #item列和j列同时有值，item列已经评分的item值集合，作为向量
            v2=dataMat[overLap, j] #item列和j列同时有值，对应的j列的值集合，作为向量
            similarity = simMeas(v1,v2)
        print('user %d noscore_columnIndex %d and isscore_columnIndex %d similarity is: %f' % (user,item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal #item的打分

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#nonzero：返回2个array,array1存放dataMat内不为0的行索引，array2存放不为0的列索引 。矩阵内user行，列为0的列索引
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:#user评分为0的列: 1,2
        estimatedScore = estMethod(dataMat, user, simMeas, item) #评分的估计分数。默认预估standEst,余弦距离
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

# data=[[1,2,3],
#       [4,5,6],
#       [7,8,9]]
# mat1=mat(data)
# v1=mat1[:,0]
# v2=mat1[:,2]
# num = float(v1.T * v2)#v1*v2
# num=la.norm(v1)
# print(num,(1+16+49)**0.5)
# denom = la.norm(v1) * la.norm(v2)

data=[
 [4,4,0,2,2],
 [4,0,0,3,3],
 [4,0,0,1,1],
 [1,1,1,2,0],
 [2,2,2,0,0],
 [1,1,1,0,0],
 [5,5,5,0,0]]
# myDat=mat(loadExData())
# myDat[0,1]=myDat[0,0]=myDat[1,0]=myDat[2,0]=4
# myDat[3,3]=2

myDat=mat(data)
aa=recommend(myDat,2) #user=2
# print(aa)

# U,sigma,VT=linalg.svd([[1,1,3],[7,7,2]])
# print(U,'--',sigma,'----',VT)






