from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  # 将数据拆开返回一个列表
        fltLine = list((map(float, curLine)))  # 将列表里面的数字转化为浮点类型
        dataMat.append(fltLine)  # 添加到列表之中
    return dataMat  # 返回一个列表

def createDataSet(mean, cov, m = 10):
    return np.random.multivariate_normal(mean, cov, m)


# 计算欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # 计算欧氏距离

# 随机构造仔质心
def randCent(dataSet, k):
    print(k)
    n = shape(dataSet)[1]  # 获取样本点的的维数
    centroids = mat(zeros((k, n)))  # zeros[k,n]构造一个k个元素，每个元素有n个0的列表
    for j in range(n):  # 随机生成一个最小向量与最大向量之间的一个向量
        minJ = min(dataSet[:, j])  # 获取所有样本点, 第J维度的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)  # 第J列的最大在减去第J列的最小值, 获取最大值与最小值之间的范围
        centroids[:, j] = minJ + rangeJ*random.rand(k, 1)  # 随机生成质心的第J维的取值
    return centroids  # 返回随机矩阵

# kMeans 算法
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 获取样本点的个数
    clusterAssment = mat(zeros((m,2)))  # 创造一个m行, 维度为2的矩阵, 每个元素的全零变量
                                        # 第一列存簇索引值，第二列存当前点到簇质心的距离
    centroids = createCent(dataSet, k)  # 随机创建k个簇心
    clusterChanged = True  # 创建标注标量，用来达到条件就终止循环
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历每个样本
            minDist = inf   # 初始样本到簇质心的距离  nif 表示正无穷；-nif表示负无穷
            minIndex = -1   # 初始化最小样本点的下标
            for j in range(k):  # 遍历每个簇质心
                distJI = distMeas(centroids[j,:], dataSet[i,:])  # 计算样本点到簇质心的距离
                if distJI < minDist:    # 寻找距离该样本点最近的簇质心
                    minDist = distJI
                    minIndex = j   # 将样本分配到距离最小的质心那簇
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # 如果样本分配结果发生变化，更改标志变量
            # 将结果存入clusterAssment
            clusterAssment[i,:] = minIndex, minDist**2  # 将该点距离其最近簇心的下标和距离记录到矩阵之中
        for cent in range(k):  # 遍历每个簇质心
            # 找到每个簇质心对应的样本
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取所有距离最近的质心为当前质心的样本点
            centroids[cent,:] = mean(ptsInClust, axis=0)  # 计算这些样本的均值，作为该簇的新质心
    return centroids, clusterAssment  # 返回质心 以及 样本点与质心之间的质心

# 可分类结果
def plot(dataSet,centValue):

    x1=dataSet[:,0]
    x2=dataSet[:,1]
    fig=plt.figure('k均值算法')
    ax=fig.add_subplot(111)

    ax.scatter(list(x1),list(x2),s=15,c='red',marker='s')
    ax.scatter(list(centValue[:,0]), list(centValue[:,1]), s=50, c='green', marker='x')
    plt.show()


if __name__ == '__main__':
    mean1 = [1, 4]
    cov1 = 0.1 * np.eye(2)

    mean2 = [1, 1]
    cov2 = 0.1 * np.eye(2)

    mean3 = [4, 1]
    cov3 = 0.1 * np.eye(2)

    mean4 = [4, 4]
    cov4 = 0.1 * np.eye(2)

    x1 = createDataSet(mean1, cov1, 30)
    x2 = createDataSet(mean2, cov2, 30)
    x3 = createDataSet(mean3, cov3, 30)
    x4 = createDataSet(mean4, cov4, 30)

    dataset = np.vstack((x1, x2))
    dataset = np.vstack((dataset, x3))
    dataset = np.vstack((dataset, x4))

    cenValue, clusterList = kMeans(dataset, 4)
    plot(dataset, cenValue)

