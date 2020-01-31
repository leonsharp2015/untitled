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

#矩阵的svd分解:A
data2=[[2,4,9],
      [1,3,12],
      [8,0,8],
      [3,2,5]]
# U2,sigma2,VT2=linalg.svd(data2)
U,sigma,VT=linalg.svd(mat(data2))#4*4,4*3,3*3 U*U.T=E,VT*VT.T=E

m_sigma=zeros((U.shape[1],sigma.shape[0]))#建立4*3的对角矩阵
m_sigma[:sigma.shape[0],:sigma.shape[0]]=diag(sigma)
#检验
A2=dot(dot(U,m_sigma),VT)

#两个矩阵元素是否相近
#只取U的3*3行,可以近似得到原矩阵
t=allclose(data2, dot(dot(U[:, :3], diag(sigma)), VT)) #U2[:, :3] * sigma2或者U[:, :3] * diag(sigma)
#取U的4*4行 dot(dot(U, m_sigma), VT)

#选取不同的奇异值，重构矩阵
dim=2
dim_sig = mat(eye(dim) * sigma[:dim])
redata = U[:,:dim] * dim_sig * VT[:dim,:]

mat1=mat(data2)
xformedItems = mat1.T * U[:, :dim] * dim_sig.I #？？？？数据集降维


# X_svd = dot(U, m_sigma)
# print(X_svd)

#pca降维
data3=[[-1,-1,0,2,0],
       [-2,0,0,1,1]]
mat3=mat(data3)
A=0.2*mat3*mat3.T
eigVals,w =linalg.eig(A)#w按列存储特征向量(e1,e2,e3,...)
p=w.T #p为特征向量按行排列的矩阵
# print(p*A*p.T)
y=p[:1,:]*mat3 #px降维
print(y)

























