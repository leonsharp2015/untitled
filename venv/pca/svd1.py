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

#ok
# data=[[2,4],
#       [1,3],
#       [0,0],
#       [0,0]]
# mat1=mat(data)#4*2
# U,sigma,VT=linalg.svd(mat1) #4*4,4*2,2*2
# # m_sigma=mat([[5.4649857,0],[0,0.36596619],[0,0],[0,0]])
# m_sigma=zeros((U.shape[1],sigma.shape[0]))
# m_sigma[:sigma.shape[0],:sigma.shape[0]]=diag(sigma)
# result=dot(dot(U,m_sigma),VT.T)
# print(result)

#ok
# data2=[[2,4,9],
#       [1,3,12],
#       [8,0,8],
#       [3,2,5]]
# mat1=mat(data2)#4*3
# U,sigma,VT=linalg.svd(mat1) #4*4,4*3,3*3
# # m_sigma=mat([[19.39838809,0,0],[0,6.41821987,0],[0,0,1.87323072],[0,0,0]])
# m_sigma=zeros((U.shape[1],sigma.shape[0]))
# m_sigma[:sigma.shape[0],:sigma.shape[0]]=diag(sigma)
# result=dot(dot(U,m_sigma),VT)
# print(result)

#ok
# data2=[[1,0,1],
#       [-1,-2,0],
#       [0,1,-1],
#       [0,2,1]]
# mat1=mat(data2)#4*3
# U,sigma,VT=linalg.svd(mat1) #4*4,4*3,3*3
# m_sigma=zeros((U.shape[1],sigma.shape[0]))
# m_sigma[:sigma.shape[0],:sigma.shape[0]]=diag(sigma)
# result=dot(dot(U,m_sigma),VT)
# print(result)


# data3=[[7,10],
#        [12,1],
#        [21,4]]
# mat1=mat(data3)#3*2
# U,sigma,VT=linalg.svd(mat1)#3*3,3*2,2*2 ,U*U.T=E,VT*VT.T=E
# m_sigma=mat([[26.12872811,0],[0,8.26375019],[0,0]])
# print(U*m_sigma*VT)

#方阵按特征向量分解：A=w*sigma*w.I
# data1=[[4,2,-5],[6,4,-9],[5,3,-7]]
# # data1=[[4,2,-5],[2,7,-9],[-5,-9,6]] #对称矩阵
# A=mat(data1)
# #w 每个特征向量的矢量范数=1
# #如果是实对称矩阵,那么它的不同特征值的特征向量必然正交,w.I=w.T
# #w1 = la.norm(w[:,2]) # |a|矢量范数
# #特征向量：
# #r1=[0.57735027,0.57735027,0.57735027]
# #r2=[-0.26726123,-0.80178373,-0.53452248]
# #r3=[0.26726125,0.80178372,0.53452249]
# eigVals,w =linalg.eig(A)#特征值约等于[1,0,0],标准化特征向量所张成的n×n维矩阵：U=[r1,r2,r3] 即r1=w[:,0],r2=w[:,1],r3=w[:,2]
# sigma=diag(eigVals) #特征值转为矩阵 sigma=mat(eye(3)*eigVals[:3])
# A2=w*sigma*w.I #任何方阵可以分解为n个特征向量所张成的n×n维矩阵、Σ为这n个特征值为主对角线的n×n维矩阵、I为矩阵的逆

#矩阵的svd分解
# data1=[[ -9.,   3.,  -7.],
#        [  4.,  -8.,  -1.],
#        [ -1.,   6.,  -9.],
#        [ -4., -10.,   2.]]

# data1=[[1,0,1],
#       [-1,-2,0],
#       [0,1,-1],
#       [0,2,1]]
# u, s, vh=linalg.svd(data1)#4*4,4*3,3*3

# d2=dot(u[:, :3]*s,vh) #4*3
# print(u[:, :3])
# print(u[:, :3])
# print(s)
# print('---------')
# print(u[:, :3]*s)

# [[-0.53815289  0.67354057 -0.13816841 -0.48748749]
#  [ 0.40133556  0.1687729   0.78900752 -0.43348888]
#  [-0.59291924  0.04174708  0.59180987  0.54448603]
#  [ 0.44471115  0.71841213 -0.09020922  0.52723647]]

#  [16.86106528 11.07993065  7.13719934]

#  [[ 0.31212695 -0.760911    0.56885078]
#  [-0.74929793 -0.56527432 -0.3449892 ]
#  [ 0.58406282 -0.31855829 -0.74658639]]

#ok
# m_sigma=array([[16.86106528,0,0],
#     [0,11.07993065,0],
#     [0,0,7.13719934],
#     [0,0,0]])
# m_sigma=zeros((u.shape[1],s.shape[0]))
# m_sigma[:s.shape[0],:s.shape[0]]=diag(s)
# B=dot(dot(u,mat(m_sigma)),vh)
# print(B)
# t=allclose(data1, dot(u[:, :3] * s, vh)) #两个矩阵元素是否相近
# print(dot(u[:, :3] * s, vh))
# print(dot(u, dot(m_sigma, vh)))

#ok
# m_sigma=zeros((4,3))
# m_sigma[:3, :3] = diag(s)
# B=dot(dot(u,mat(m_sigma)),vh.T)


# u1=array([  [-1,0,-0.1,1],
#             [ 0.4,0.1,0.7,1],
#             [-0.5,0.1,0.5,1],
#             [ 0.4,0.7,0.1,1]])
# s1=array([10,5,6])
# v1=array([
#             [ 0.3,-0.7, 0.5],
#             [-0.7,-0.5,-0.3],
#             [0.5,-0.3,-0.7]
#         ])
# print(u1[:, :3]*s1) #每个元素分别相乘
# print(dot(u1[:, :3]*s1,v1))#矩阵相乘


A = mat([[1, 2, 3], [4, 5, 6]])
U, Sigma, VT = linalg.svd(A)
print("U", U)
print("Sigma", Sigma)
print("VT", VT)
Sigma[1] = 0  # 降维
print("Sigma", Sigma)

S = zeros((2, 3))
S[:2, :2] = diag(Sigma)
print('S:',S)
print("A conv:", dot(dot(A.T, U), S))  # 原始数据转到低维A.T*U*S
print("A':", dot(dot(U, S), VT)) # 恢复原始维度

