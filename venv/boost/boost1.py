from numpy import *
def loadSimpData():
    datMat = matrix([[ 1. ,2.1],
        [ 2. , 1.1],
        [ 1.3, 1. ],
        [ 1. , 1. ],
        [ 2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels



def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):  # mat,列索引，列步长值，lt或者gt:对传递过来的列，判断每列的值是否lt或者gt列步长值threshVal。满足条件则将对应的列值定为－1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0 #等于也是－1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr);
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)#5,2
    numSteps = 10.0;
    bestStump = {};
    bestClasEst = mat(zeros((m, 1)))
    minError = inf  # init error sum, to +infinity

    for i in range(n):  # loop over all dimensions
        rangeMin = dataMatrix[:, i].min();#第0个特征,第1个特征最小值
        rangeMax = dataMatrix[:, i].max();
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):  # loop over all range in current dimension
            for inequal in ['lt', 'gt']:  # go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)#步长值
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)  # call stump classify with i, j, lessThan
                # print(dataMatrix[:,i],inequal,threshVal,predictedVals,'******')
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # calc total error multiplied by D

                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy() #计算得到的分类
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal 权重 5*1
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        # print ("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha #{'dim': 0, 'alpha': 0.6931471805599453, 'thresh': 1.3, 'ineq': 'lt'}
        weakClassArr.append(bestStump)                  #store Stump Params in Array
         #classLabels实际分类，classEst计算所得分类
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy 5*1 :a
        D = multiply(D,exp(expon))                              #Calc New D for next iteration 新的权重D  D*e^a   5*1
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst #每个点的类别估计值
        # print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        # print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):#测试数据
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)

datMat,classLabels=loadSimpData()
# D=mat(ones((5,1))/5)
# bestStump, minError, bestClasEst=buildStump(datMat,classLabels,D)
# print(bestStump,'****',minError,'***',bestClasEst)

#训练得到分类器
classifierArr,aggClassEst=adaBoostTrainDS(datMat,classLabels,9)

test_c=adaClassify([0,0],classifierArr)
print(test_c)








