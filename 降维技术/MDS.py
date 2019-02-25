import numpy as np
from numpy import linalg as la


# 加载数据
def loadDataSet():
    D = np.array([[101,243,354,421,541],
                  [234,241,432,521,123],
                  [902,259,102,921,416],
                  [125,251,255,125,186],
                  [903,560,375,125,654],
                  [124,854,731,643,146]])
    return D


# 计算两个样本点之间的欧氏距离
def calEuclDist(inX,inY):
    n = len(inX)
    distance = 0
    for i in range(n):
        distance += (inX[i] - inY[i])**2
    return np.sqrt(distance)


# 计算距离矩阵
def calDistMatrix(dataSet):
    m = dataSet.shape[0]
    distMatrix = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            distMatrix[i,j] = calEuclDist(dataSet[i,:], dataSet[j,:])
    return distMatrix


# MDS降维算法
def MDS(dataSet, lowDimen=2):
    '''
    :param dataSet: 数据集
    :param lowDimen: 降维多少维
    :return: 降维之后的矩阵
    '''
    distMatrix = calDistMatrix(dataSet)  # 获取距离矩阵
    m = distMatrix.shape[0]  # 获取样本点的个数
    B = np.zeros((m, m))  # 内积矩阵B
    distMatrix2 = distMatrix ** 2  # 计算距离矩阵的平方
    H = np.eye(m) - 1/m  # 获取辅助矩阵
    # 计算bij = -0.5*(distij - dist^2(i.) - dist^2(.j) + dist^2(..))
    B = -0.5 * np.dot(np.dot(H, distMatrix2), H)  # 公式(10.10)

    # 方法一
    U, Sigma, VT = la.svd(B)  # 矩阵分解
    ChildSigma = np.diag(np.sqrt(Sigma[:lowDimen]))  # 获取前lowDimen个奇异值
    Z1 = np.dot(ChildSigma, VT[:lowDimen, :]).T  # 获取降维之后的矩阵
    calSigmaEnergy(Sigma, lowDimen)

    # 方法2
    # 计算矩阵的特征值eigVal, 以及特征向量eigVec, 不用对eigVal再排序了
    eigVal, eigVec = np.linalg.eig(B)
    Z = np.dot(eigVec[:, :lowDimen], np.diag(np.sqrt(eigVal[:lowDimen])))  # 生成降维之后的矩阵,对应公式(10.12)
    calSigmaEnergy(eigVal, lowDimen)
    return Z, Z1

# 计算降维后的能量占比
def calSigmaEnergy(Sigma, index):
    Sig = Sigma**2  # 处理Sigma
    totalEnergy = sum(Sig)  # 计算总能量
    childEnergy = sum(Sig[:index])  # 计算钱i个奇异值的总能量
    print('总能量占比%s'%((childEnergy/totalEnergy)*100),'%')

if __name__ == '__main__':
    dataSet = loadDataSet()
    m = dataSet.shape[0]
    newDataSet1, newDataSet2 = MDS(dataSet, 2)
    print('newDataSet1=',newDataSet1)
    print('newDataSet2=',newDataSet2)
    distMatrix = calDistMatrix(dataSet)
    print('original distance', '\tnew distance')
    for i in range(m):
        for j in range(i + 1, m):
            old_distance = distMatrix[i, j]  # 降维之前矩样本点之间的距离
            # np.linalg.norm　计算范数
            new_distance = np.linalg.norm(newDataSet1[i] - newDataSet1[j])  # 降维之后样本点的距离
            print(np.str("%.4f"%old_distance), '\t\t', np.str("%.4f" % new_distance))

    print('original distance', '\tnew distance')
    for i in range(m):
        for j in range(i + 1, m):
            old_distance = distMatrix[i, j]  # 降维之前矩样本点之间的距离
            # np.linalg.norm　计算范数
            new_distance = np.linalg.norm(newDataSet2[i] - newDataSet2[j])  # 降维之后样本点的距离
            print(np.str("%.4f" % old_distance), '\t\t', np.str("%.4f" % new_distance))
