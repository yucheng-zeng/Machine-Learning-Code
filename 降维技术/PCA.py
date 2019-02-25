from numpy import *
import matplotlib.pyplot as plt

# 加载数据, 将原始数据转换为矩阵型
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]  # 将每行数据切分, 得到了是列表，每个元素是字符串
    datArr = [list(map(float,line)) for line in stringArr]  # 把每行字符串映射为浮点型数字
    #print('datArr=',datArr)
    return mat(datArr)  # 返回的是矩阵型数据集

# 将数据降维
def pca(dataMat, topNfeat=9999999):
    '''
    :param dataMat: 矩阵型原始数据
    :param topNfeat: 保留的特征个数
    :return:
    '''
    totalVar = calMatVar(dataMat)  # 计算样本总方差

    # print(totalVar)
    meanVals = mean(dataMat, axis=0)  # 所有行对应维度相加, 然后除以行数, 的到每一个维度的平均值
    #print('meanVals=', meanVals)
    meanRemoved = dataMat - meanVals  # 原数据集移除均值
    #print('meanRemoved=',meanRemoved)
    covMat = cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    #print('covMat=',covMat)
    # 计算特征值eigVals, 计算特征向量eigVects
    eigVals, eigVects = linalg.eig(mat(covMat))
    #print('eigVals=',eigVals)
    #print('eigVects=', eigVects)
    eigValInd = argsort(eigVals)  # 排序, 从小到达排序
    #print('eigValInd=',eigValInd)
    # 后面的-1代表的是将值倒序，原来特征值从小到大，现在从大到小
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 获取指定个特征值最大的下标
    #print('eigValInd=',eigValInd)
    #print 'eigValInd=',eigValInd
    redEigVects = eigVects[:, eigValInd]  # 获取前eigValInd个特征值最大的特征向量
    #print('redEigVects=',redEigVects)
    lowDDataMat = meanRemoved * redEigVects  # 将数据转换到新空间中, 降维之后的数据集

    lowTotalVar = calMatVar(lowDDataMat)  # 计算样本经过降维以后的方差

    print('方差百分比=',((lowTotalVar/totalVar)*100),'%')
    #print('lowDDataMat=', lowDDataMat)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 降维后的数据再次映射到原来空间中，用于与原始数据进行比较
    return lowDDataMat, reconMat


# 计算矩阵的总方差
def calMatVar(targetMat):
    m, n = shape(targetMat)  # 获取矩阵的行数, 列数
    lowTotalVar = 0  # 记录矩阵总的方差
    for i in range(n):
        CVet = targetMat[:, i]
        lowTotalVar += CVet.var() * m  # 乘以样本个数, 放大方差, 防止数据浮点数过小
    return lowTotalVar


# 绘图
def fig(dataMat,reconMat,lowDMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s = 20) #原始数据集，标记为三角形
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s = 20,c = 'red')#重构后的数据，标记为圆形

    # 以下两天曲线是降维之后的曲线
    #lowDMaty = mat(zeros((lowDMat.shape[0], 1)))
    #ax.scatter(lowDMat[:,0].flatten().A[0],lowDMaty[:,0].flatten().A[0],marker='*',s = 20,c = 'y')#重构后的数据，标记为圆形
    #ax.scatter(reconMat[:, 0].flatten().A[0], lowDMaty[:, 0].flatten().A[0], marker='^', s=20, c='b')  # 重构后的数据，标记为圆形
    plt.show()


# 实例
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])  # 计算平均值
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  # 设置空值为平均值
    return datMat

if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    #print('dataMat=',dataMat)
    lowDMat, reconMat = pca(dataMat,1)
    #print('reconMat=',reconMat)
    #print('lowDMat=',lowDMat)
    fig(dataMat, reconMat, lowDMat)

    '''
    dataMat = replaceNanWithMean()
    lowDMat, reconMat = pca(dataMat, 3)
    '''

