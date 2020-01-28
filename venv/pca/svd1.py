from numpy import  *
U,sigma,VT=linalg.svd([[1,1,3],[7,7,2]])
print(U,'--',sigma,'----',VT)
