'''
好处
1. 与K-means方法相比，DBSCAN不需要事先知道要形成的簇类的数量。
2. 与K-means方法相比，DBSCAN可以发现任意形状的簇类。
3. 同时，DBSCAN能够识别出噪声点。
4.DBSCAN对于数据库中样本的顺序不敏感，即Pattern的输入顺序对结果的影响不大。但是，对于处于簇类之间边界样本，可能会根据哪个簇类优先被探测到而其归属有所摆动。
缺点
1. DBScan不能很好反映高维数据。
2. DBScan不能很好反映数据集以变化的密度。
'''


import numpy as np
import matplotlib.pyplot as plt
import math
import time

UNCLASSIFIED = False
NOISE = 0


# 从文件读入数据集
def loadDataSet(fileName, splitChar='\t'):
    '''
    :param fileName: 文件名
    :param splitChar: 数据分割符
    :return: 数据集
    '''

    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curline = line.strip().split(splitChar)
            fltline = list(map(float, curline))
            dataSet.append(fltline)
    return dataSet


def dist(a, b):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    return math.sqrt(np.power(a - b, 2).sum())


def eps_neighbor(a, b, eps):
    """
    输入：向量A, 向量B
    输出：是否在eps范围内
    """
    return dist(a, b) < eps


# 寻找以目标点为圆心,半径为eps的园范围内的点
def region_query(data, pointId, eps):
    '''
    :param data: 数据集
    :param pointId: 查询点id
    :param eps: 半径大小
    :return: 在eps范围内的点的id
    '''
    nPoints = data.shape[1]  # 获取样本点的个数
    seeds = []  # 用于记录所有在以点data[:, pointId]为圆心, 半径为eps的园范围内的点
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):  # 判断是否满足条件
            seeds.append(i)  # 添加满足条件的点
    return seeds  # 返回满足条件的点的集合


# 扩张簇
def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    '''
    :param data: 数据集
    :param clusterResult: 分类结果
    :param pointId: 待分类点id
    :param clusterId: 簇id
    :param eps: 半径大小
    :param minPts: 最小点个数
    :return: 能否成功分类
    '''

    seeds = region_query(data, pointId, eps)  # 寻找以目标点为圆心,半径为eps的园范围内的点
    if len(seeds) < minPts:  # 不满足minPts条件的点不是核心对象点, 且有可能为噪声点
        clusterResult[pointId] = NOISE  # 将该点设置为噪声点
        return False

    # 满足条件, 该点是核心对象点
    else:
        clusterResult[pointId] = clusterId  # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId  # 将所有在以点data[:, pointId]为圆心, 半径为eps的园范围内的点划分到该簇

        while len(seeds) > 0:  # 持续扩张
            currentPoint = seeds[0]  # 以seeds[0]为中心点重复上述过程
            queryResults = region_query(data, currentPoint, eps)  # 寻找以目标点为圆心,半径为eps的园范围内的点
            if len(queryResults) >= minPts:  # 满足条件, 该点是核心对象点
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:  # 若果该范围之内的点还没有被划分
                        seeds.append(resultPoint)  # 增加该点到seed之中
                        clusterResult[resultPoint] = clusterId  # 将该点划分到当前簇
                    elif clusterResult[resultPoint] == NOISE:  # 若果该点是被误划分为噪声点
                        seeds.append(resultPoint)  # 增加该点到seed之中
                        clusterResult[resultPoint] = clusterId  # 将该点划分到当前簇
            seeds = seeds[1:]  # 删除seed第一个元素
        return True

# DBSCAN 算法
def dbscan(data, eps, minPts):
    '''
    :param data: 数据集
    :param eps: 半径大小
    :param minPts: 最小点个数
    :return: 分类簇id
    '''

    clusterId = 1  # 记录簇个数
    nPoints = data.shape[1]  # 获取样本点的个数
    clusterResult = [UNCLASSIFIED] * nPoints  # 创建一个数值为布尔型的列表, 用于记录样本点是否已经被划分到一个簇之中
    for pointId in range(nPoints):  # 遍历所有样本点
        point = data[:, pointId]  # 获取第pointId个样本点
        if clusterResult[pointId] == UNCLASSIFIED:  # 若果该样本点还没有被划分到一个簇之中
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):  # 寻找簇
                clusterId = clusterId + 1  # 簇个数+1
    return clusterResult, clusterId - 1

# 绘图
def plotFeature(data, clusters, clusterNum):
    matClusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(clusterNum + 1):
        colorSytle = scatterColors[i % len(scatterColors)]  # 获取一种颜色
        subCluster = data[:, np.nonzero(matClusters[:, 0].A == i)[0]]  # 获取所有被划分到簇i的点
        if i == 0:
            labels = 'NOISE'
        else:
            labels = 'cluster '+str(i)
        ax.scatter(subCluster[0, :].flatten().A[0], subCluster[1, :].flatten().A[0], c=colorSytle, s=50, label=labels)
    plt.legend(loc='upper right')
    plt.show()

def main():
    start = time.clock()  # 记录开始时间
    dataSet = loadDataSet('testSet.txt', splitChar='\t')  # 加载数据
    print(dataSet)
    dataSet = np.mat(dataSet).transpose()  # 预处理数据
    print(dataSet)
    clusters, clusterNum = dbscan(dataSet, 2, 15)  # DBSCAN算法
    print("cluster Numbers = ", clusterNum)
    print('clusters=%s'%clusters)
    end = time.clock()  # 记录算法结束时间
    print('finish all in %s' % str(end - start))
    plotFeature(dataSet, clusters, clusterNum)  # 绘图

if __name__ == '__main__':
    main()

