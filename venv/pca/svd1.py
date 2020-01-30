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


def svdEst(dataMat, user, simMeas, item):#svd矩阵分解
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4]) #对角矩阵 arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #构造为低维矩阵。包含大多数原矩阵的量。create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#nonzero：返回2个array,array1存放dataMat内不为0的行索引，array2存放不为0的列索引 。矩阵内user行，列为0的列索引
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:#user评分为0的列: 1,2
        estimatedScore = estMethod(dataMat, user, simMeas, item) #评分的估计分数。默认预估standEst,余弦距离
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

#--------------------------------

def Mat_print(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1),
            else: print(0),
        print ('')

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch14/0_5.txt').readlines():#32列
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print ("****original matrix******")
    Mat_print(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ("****reconstructed matrix using %d singular values******" % numSV)
    Mat_print(reconMat, thresh)



#范数
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

# data=[
#  [4,4,0,2,2],
#  [4,0,0,3,3],
#  [4,0,0,1,1],
#  [1,1,1,2,0],
#  [2,2,2,0,0],
#  [1,1,1,0,0],
#  [5,5,5,0,0]]
# myDat=mat(data)
# items_score=recommend(myDat,2) #user=2
# print(items_score)

#svd奇异值的能量
# data=[     [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
#            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
#            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
#            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
#            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
#            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
#            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
#            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
#            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
#            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
#            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# mat1=mat(data)#11*11
# U,sigma,VT=linalg.svd(mat1) #11*11
# sig2=sigma**2
# s_all=sum(sig2)
# s_1=sum(sig2[:2])

# Sig4 = mat(eye(4) * sigma[:4])  # 4对角矩阵 arrange Sig4 into a diagonal matrix
# xformedItems = mat1.T * U[:, :4] * Sig4.I #数据集降维
# redata = U[:,:4] * Sig4 * VT[:4,:]   # 重构

# data=[[2,4],
#       [1,3],
#       [0,0],
#       [0,0]]
# mat1=mat(data)#4*2
# U,sigma,VT=linalg.svd(mat1) #4*4,4*2,2*2
# m_sigma=mat([[5.4649857,0],[0,0.36596619],[0,0],[0,0]])
# result=U*m_sigma*VT.T
# print(U)

#????
# data2=[[2,4,9],
#       [1,3,12],
#       [8,0,8],
#       [3,2,5]]
# mat1=mat(data2)#4*3
# U,sigma,VT=linalg.svd(mat1) #4*4,4*3,3*3
# m_sigma=mat([[19.39838809,0,0],[0,6.41821987,0],[0,0,1.87323072],[0,0,0]])
# print(U*m_sigma*VT.T)
# print(U,sigma,VT)

#?????
data2=[[1,0,1],
      [-1,-2,0],
      [0,1,-1]]
mat1=mat(data2)#3*3
U,sigma,VT=linalg.svd(mat1,full_matrices=0) #3*3,3*3,3*3
m_sigma=mat([[2.46050487,0,0],[0,1.69962815,0],[0,0,0.23912328]])
# print(U*m_sigma*VT.T)
# print(VT*VT.T)
print(U,sigma,VT)


# data3=[[7,10],
#        [12,1],
#        [21,4]]
# mat1=mat(data3)#3*2
# U,sigma,VT=linalg.svd(mat1)#3*3,3*2,2*2
# m_sigma=mat([[26.12872811,0],[0,8.26375019],[0,0]])
# # print(U,sigma,VT)
# print(U*m_sigma*VT.T)

