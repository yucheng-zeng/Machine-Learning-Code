from numpy import *
import matplotlib.pyplot as plt
import copy

'''
这个算法对初值太敏感了, 特别是对均值向量mu的初值
'''
# 加载数据, 预处理数据
def loadData(filename):
    dataSet = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = list(map(float, curLine))  # 将数据映射为浮点型数据
        dataSet.append(fltLine)
    return dataSet


# 高斯分布的概率密度函数
def prob(x, mu, sigma):
    n = shape(x)[1]  # 获取样本点的维数
    expOn = float(-0.5 * (x - mu) * (sigma.I) * ((x - mu).T))
    divBy = pow(2 * pi, n / 2) * pow(linalg.det(sigma), 0.5)
    return pow(e, expOn) / divBy


# EM算法
def EM(dataMat, clusterNum = 3, maxIter=500, valve=1e-5):
    '''
    :param dataMat: 数据集
    :param clusterNum: 簇个数
    :param maxIter: 最大迭代次数
    :return: 样本点的后验分布
    '''
    m, n = shape(dataMat)  # 获取数据集的样本点个数以及样本点的维度
    #alpha = [1 / 3, 1 / 3, 1 / 3]  # 初始化各高斯混合成分参数
    alpha = []  # 记录各高斯混合成分参数
    #mu = [dataMat[:int(m/3), :], dataMat[int(m/3):int((m*2)/3), :], dataMat[int((m*2)/3):, :]]  #
    #mu = [dataMat[int(m/3), :], dataMat[int(m*2/3), :], dataMat[int(m-1), :]]  # 初始化mu
    mu= []  # 记录均值向量mu

    # 高斯混合模型是对初值敏感的模型, mu的设置一定要保证随机选
    for i in range(clusterNum):
        alpha.append(1/clusterNum)    # 初始化各高斯混合成分参数
        index = int(random.uniform(0, m))  # 随机生成一个整数
        mu.append(dataMat[index, :])  # 初始化mu
    #print('alpha=%s'%alpha)
    #print('mu=%s'%mu)
    sigma = [mat([[0.1, 0], [0, 0.1]]) for x in range(clusterNum)]  # 初始化协方差矩阵
    gamma = mat(zeros((m, clusterNum)))  # 初始化后验概率gamma
    old_sumGamma = []  # 记录跌代前的每个样本Gamma的总和
    sumGamma = mat(zeros((1,clusterNum)))  # 记录的每个样本Gamma的总和
    for i in range(maxIter):  #
        # E step
        for j in range(m):  # 遍历所有的样本点
            sumAlphaMulP = 0
            for k in range(clusterNum):
                # 计算样本点j, 第k个高斯混合成分生成的后验概率
                gamma[j, k] = alpha[k] * prob(dataMat[j, :], mu[k], sigma[k])  # 第k个高斯混合成分生成的后验概率的分子
                sumAlphaMulP += gamma[j, k]  # 求和得出分母分母
            for k in range(clusterNum):
                gamma[j, k] = gamma[j, k]/sumAlphaMulP  # 计算样本点j, 第k个高斯混合成分生成的后验概率

        old_sumGamma = copy.deepcopy(sumGamma)  # 记录跌代前的每个样本Gamma的总和
        sumGamma = sum(gamma, axis=0)  # 计算所有样本的高斯混合成分后验概率对应的维度的和
        # 计算迭代前后sumGamma的差距, 若果小于阀值, 退出
        distance = 0
        for k in range(clusterNum):
            distance += (old_sumGamma.tolist()[0][k] - sumGamma.tolist()[0][k])**2
        distance = sqrt(distance)
        if distance < valve:
            print('Iter=%s'%i)
            break
        # M step
        for k in range(clusterNum):
            mu[k] = mat(zeros((1, n)))
            sigma[k] = mat(zeros((n, n)))
            # 更新mu
            for j in range(m):
                mu[k] += gamma[j, k] * dataMat[j, :]
            mu[k] /= sumGamma[0, k]
            # 更新sigma
            for j in range(m):
                sigma[k] += gamma[j, k] * (dataMat[j, :] - mu[k]).T * (dataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]
            # 更新alpha
            alpha[k] = sumGamma[0, k] / m
    # print(mu)
    return gamma


# 初始化聚类簇
def initCentroids(dataMat, k):
    numSamples, dim = dataMat.shape  # 获取数据集的样本点个数以及样本点的维度
    centroids = zeros((k, dim))  # 初始化k个, 维度为dim的聚类簇的坐标
    for i in range(k):  # 所有聚类簇
        index = int(random.uniform(0, numSamples))  # 随机生成一个整数
        centroids[i, :] = dataMat[index, :]  # 初始化聚类簇
    return centroids


# 高斯聚类
def gaussianCluster(dataMat, clusterNum = 3):
    m, n = shape(dataMat)  # 获取数据集的样本点个数以及样本点的维度
    if clusterNum > m:
        print('the number of clusters is illegal, it should be less than %s'%m)
        return
    # 每个样本的所属的簇，以及分到该簇对应的响应度

    centroids = initCentroids(dataMat, clusterNum)  # 初始化聚类簇
    clusterAssign = mat(zeros((m, n)))  # 初始化簇分配矩阵
    gamma = EM(dataMat, clusterNum)  # 计算后验概率
    # 确定每个样本的簇标记
    for i in range(m):
        # argmax返回矩阵最大值所在下标, amx返回矩阵最大值
        clusterAssign[i, :] = argmax(gamma[i, :]), amax(gamma[i, :])
        #print('clusterAssign[i, :]=%s'%clusterAssign[i, :])
    # 确定聚类簇点坐标
    for j in range(clusterNum):
        # 获取所有类别标记为j的点
        pointsInCluster = dataMat[nonzero(clusterAssign[:, 0].A == j)[0]]
        #print('%s,pointsInCluster=%s'%(j,pointsInCluster))
        centroids[j, :] = mean(pointsInCluster, axis=0)  # 计算这些点的平均值作为聚类簇的坐标
    return centroids, clusterAssign


def showCluster(dataMat, k, centroids, clusterAssment):
    numSamples, dim = dataMat.shape
    if dim != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large!")
        return 1

        # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()

if __name__ == '__main__':
    dataMat = mat(loadData('watermelon4.txt'))
    #print(dataMat)
    clusterNum = 2  # 簇个数
    centroids, clusterAssign = gaussianCluster(dataMat, clusterNum)
    #print(clusterAssign)
    showCluster(dataMat, clusterNum, centroids, clusterAssign)