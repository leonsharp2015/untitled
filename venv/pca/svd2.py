from numpy import  *
from numpy import linalg as la

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

#edv降维pca
#5条2维数据，要写成2*5矩阵
# X=[[-1,-1,0,2,0],
#     [-2,0,0,1,1]]
# X=mat(X)
# C=0.2*X*X.T #协方差矩阵
# eigVals,w =linalg.eig(C)# w按列存储特征向量(e1,e2,e3,...)
# p=w.T #p为特征向量按行排列的矩阵.一组基按行组成
# # print(p*C*p.T) #p*A*p.T为对角矩阵，值为特征值2,4
# y=p[:1,:]*X #px降维.k个基乘以X就是X由N维降为k维
# print('y1:',y)
#
# #也可以
# X2=[[-1,-2],
#     [-1,0],
#     [0,0],
#     [2,1],
#     [0,1]]
# X=mat(X2)
# C=0.2*X.T*X#协方差矩阵
# eigVals,w =linalg.eig(C)# w按列存储协方差矩阵的特征向量(e1,e2,e3,...)
# y=X*w[:,0] #X*e1
# print('y2:',y)


#矩阵的svd分解
# data2=[[2,4,9],
#       [1,3,12],
#       [8,0,8],
#       [3,2,5]]
# # U2,sigma2,VT2=linalg.svd(data2)
# U,sigma,VT=linalg.svd(mat(data2))#4*4,4*3,3*3 U*U.T=E,VT*VT.T=E
#
# m_sigma=zeros((U.shape[1],sigma.shape[0]))#建立4*3的对角矩阵
# m_sigma[:sigma.shape[0],:sigma.shape[0]]=diag(sigma)
# #检验
# A2=dot(dot(U,m_sigma),VT)
#
# #两个矩阵元素是否相近
# #只取U的3*3行,可以近似得到原矩阵
# t=allclose(data2, dot(dot(U[:, :3], diag(sigma)), VT)) #U2[:, :3] * sigma2或者U[:, :3] * diag(sigma)
# #取U的4*4行 dot(dot(U, m_sigma), VT)
#
# #选取不同的奇异值，重构矩阵
# dim=2
# dim_sig = mat(eye(dim) * sigma[:dim])
# redata = U[:,:dim] * dim_sig * VT[:dim,:]

#svd的降维pca
A=mat([[5,5,0,5],
      [5,0,3,4],
      [3,4,0,3],
      [0,0,5,3],
      [5,4,4,5],
      [5,4,5,5]]) #6*4
U,Sigma,V=linalg.svd(A) #6*6,6*4,4*4
#减少U、sigma的维度
U=U[:,:2] #6*2
S=zeros((2,2))
S = mat(eye(2) * Sigma[:2])#2*2
V=V[0:2,:] #2*4

low_rows=U.T*A #X=U*S*VT 得到UT*X=S*VT,即对A进行行压缩
low_col=A*V.T #即对A进行列压缩
lowA=A.T*U*S.I #由A=U*S*VT,得到V=AT*U*S.I,其中S.I.T＝S.I。也就是V表示将A压缩以后的矩阵
#可对A的新向量a1做降维
a1=mat([[5],[5],[0],[0],[0],[5]])
a1_lowdata=a1.T*U*S.I

#也可以对A的列进行压缩，先进行变型
#-----对A的列压缩:A:6*4压缩成6*2
#对A转置为B，然后通过U.T*B得到B的行压缩，也就是A的列压缩
B=A.T #4*6
U,Sigma,V=linalg.svd(B) #4*4,4*6,6*6
#减少U、sigma的维度
U=U[:,:2] #4*2
S=zeros((2,2))
S = mat(eye(2) * Sigma[:2])#2*2
V=V[0:2,:] #2*6
low_row=U.T*B #对B进行行压缩.但是由于B是A的转置，所以也就是通过U对A进行了列压缩
low_col=B*V.T #对B进行列压缩































