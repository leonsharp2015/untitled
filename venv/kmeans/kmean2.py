from numpy import *
def loadData(path):
    datalist=[]
    fr=open(path)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        fitline=list(map(float,curline))
        datalist.append(fitline)
    return datalist

def randCent(dataSet,k):
    n=dataSet.shape[1]
    centids=mat(zeros((k,n)))
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangJ=float(max(dataSet[:,j])-minJ)
        centids[:,j]=minJ+rangJ*random.rand(k,1)
    return centids

def disEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def kmeans2(dataSet,k):
    m=dataSet.shape[0]
    dataAssment=mat(zeros((m,2)))
    centids=randCent(dataSet,k)
    for i in range(m):
        v1=dataSet[i,:]
        min_distance=inf
        centid_index=0
        for j in range(k):
            c=centids[j,:]
            dis=disEclud(v1,c)
            if(dis<min_distance):
                min_distance=dis
                centid_index=j
        dataAssment[i,:]=[min_distance,centid_index]

    # #0类
    # rows_index_0=nonzero(dataAssment[:,1:2]==0)[0]
    # data_0=dataSet[rows_index_0,:]
    # new_cent_x_0=sum(data_0[:,:1])/data_0.shape[0]
    # new_cent_y_0 = sum(data_0[:, 1:2]) / data_0.shape[0]
    # #1类
    # rows_index_1=nonzero(dataAssment[:,1:2]==1)[0]
    # data_1=dataSet[rows_index_1,:]
    # new_cent_x_1=sum(data_1[:,:1])/data_0.shape[0]
    # new_cent_y_1 = sum(data_1[:, 1:2]) / data_0.shape[0]

    #通过dataAssment第二列保存的row_index,分类data.取分组data的列均值作为点坐标
    for i in range(k):
        data=dataSet[nonzero(dataAssment[:,1:2]==i)[0],:]
        centids[i,:]=mean(data,axis=0)

    return dataAssment


datalist=loadData('/Users/zhanglei/机器学习与算法/机器学习实战源代码/machinelearninginaction/Ch10/testSet.txt')
dataSet=mat(datalist)
dataAssment=kmeans2(dataSet,4)


