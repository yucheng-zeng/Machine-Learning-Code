import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as image

# 加载数据, 将原始数据转换为矩阵型
def create_data(mean, cov, m):
    '''
    :param mean:  均值矩阵
    :param cov:   协方差矩阵
    :param m:  生成数目
    :return:
    '''
    x = np.random.multivariate_normal(mean, cov, m)
    return x

def pca(dataMat, topNfeat=9999999, axis=0):
    '''
    :param dataMat: 矩阵型原始数据
    :param topNfeat: 保留的特征个数
    :param axis:  标志数据矩阵那个维度是特征
    :return:
    '''
    meanVals = np.mean(dataMat, axis=axis)  # 所有行对应维度相加, 然后除以行数, 的到每一个维度的平均值
    meanRemoved = dataMat - meanVals  # 原数据集移除均值
    covMat = np.cov(meanRemoved, rowvar=axis)  # 计算协方差矩阵
    # 计算特征值eigVals, 计算特征向量eigVects
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)  # 排序, 从小到达排序
    # 后面的-1代表的是将值倒序，原来特征值从小到大，现在从大到小
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 获取指定个特征值最大的下标
    redEigVects = eigVects[:, eigValInd]  # 获取前eigValInd个特征值最大的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 将数据转换到新空间中, 降维之后的数据集
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 降维后的数据再次映射到原来空间中，用于与原始数据进行比较
    return lowDDataMat, reconMat

# 绘图
def fig(dataMat,reconMat,lowDMat):
    ax1 = plt.subplot(221)
    ax1.scatter(lowDMat[:, 0].tolist(), lowDMat[:, 1].tolist(), c='y')
    ax1.set_title('point on hyperplane')

    ax = plt.subplot(222, projection='3d')
    ax.scatter(dataMat[:, 0].tolist(), dataMat[:, 1].tolist(), dataMat[:, 2].tolist(), c='r')  # 绘制数据点
    ax.scatter(reconMat[:, 0].tolist(), reconMat[:, 1].tolist(), reconMat[:, 2].tolist(), c='b')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_title('point on original space')
    plt.show()

if __name__ == '__main__':
    mean1 = [1, 1, 1]
    cov1 = 4 * np.eye(3)
    cov1[2][2] = 10
    dataMat = create_data(mean1, cov1, 100)
    lowDMat, reconMat = pca(dataMat, 2)
    fig(dataMat, reconMat, lowDMat)
