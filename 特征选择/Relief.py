import numpy as np
from random import randrange
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
import operator


# 计算距离
def distanceNorm(Norm, D_value):
    # initialization

    # 三种计算距离的方式
    if Norm == '1':  # 曼哈顿距离
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':  # 欧氏距离
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':  # 切比雪夫距离
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')

    return counter


# Relief 算法
def fit(features, labels, iter_ratio, k, norm):
    '''
    :param features: 数据集特征
    :param labels: 数据集标记
    :param iter_ratio: 计算轮数
    :param k: k个最近邻的点
    :param norm: 距离计算方式
    :return: 权重
    '''
    # 初始化
    n_samples, n_features = np.shape(features)  # 获取数据集样本个数以及, 以及样本点的位数
    distance = np.zeros((n_samples, n_samples))  # 创建距离矩阵
    weight = np.zeros(n_features)  # 创建权重列表
    #print(weight)
    labels = list(map(int, labels))  # 创建标记列表, 用于记录每个样本点的标记

    # 为节省时间这里只计算一半, 得到一个三角阵
    for index_i in range(n_samples):  # 遍历所有样本点
        for index_j in range(index_i + 1, n_samples):  # 计算样本点i与所有样本点的距离
            D_value = features[index_i] - features[index_j]
            distance[index_i, index_j] = distanceNorm(norm, D_value)
    distance += distance.T  # 补全矩阵

    # 开始迭代
    for iter_num in range(int(iter_ratio * n_samples)):
        index_i = randrange(0, n_samples, 1)  # 随机获取一个样本, 可以降低噪点选取率噪点
        self_features = features[index_i]  # 获取该样本的所有特征

        # 初始化
        nearHit = list()  # 初始化猜中近邻
        nearMiss = dict()  # 初始化猜错近邻
        n_labels = list(set(labels))  # 初始化标记集合
        termination = np.zeros(len(n_labels))  # 所有元素都为0的列表, 用于记录该标签的最近邻的k个点是否已经找全
        del n_labels[n_labels.index(labels[index_i])]  # 从标记集合里面删除被选中样本的标记
        for label in n_labels:  # 遍历与选中样本不同的标记
            nearMiss[label] = list()  # 每个标记对应着一个列表, 用于存储猜错近邻点
        distance_sort = list()  # 用于记录被选中点与其他点的距离, 下标 以及 标记

        # 搜寻猜中近邻点 以及 猜错近邻点
        distance[index_i, index_i] = np.max(distance[index_i])  # 将被选中样本点自身距离设置为最大值, 避免被误选为最近邻点
        for index in range(n_samples):  # 遍历所有样本
            distance_sort.append([distance[index_i, index], index, labels[index]])  # 添加被选中样本点与其他样本点的信息

        distance_sort.sort(key=lambda x: x[0])  # 按距离从小到大排序

        for index in range(n_samples):  # 遍历所有的样本点
            # 搜寻猜中近邻
            if distance_sort[index][2] == labels[index_i]:  # 若果该样本点与被选中样本点同属于一个类
                if len(nearHit) < k:  # 属于最近邻的k个
                    nearHit.append(features[distance_sort[index][1]])  # 增加该样本的下标到猜中近邻列表
                else:
                    termination[distance_sort[index][2]] = 1  # 用于标记该类的最近邻的k个点已经找全
            # 搜寻猜错近邻
            elif distance_sort[index][2] != labels[index_i]:
                if len(nearMiss[distance_sort[index][2]]) < k:   # 若果该样本点与被选中样本点不同属于一个类
                    nearMiss[distance_sort[index][2]].append(features[distance_sort[index][1]])   # 增加该样本的下标到相应的猜错近邻列表
                else:
                    termination[distance_sort[index][2]] = 1  # 用于标记该类的最近邻的k个点已经找全

            if list(map(int, list(termination))).count(0) == 0:  # 判断是否提前退出
                break

        # 更新权重, 对应公式11.4
        nearHit_term = np.zeros(n_features)  # 计算
        for x in nearHit:  # 遍历所有猜中近邻点
            num = np.power(self_features - x, 2)
            nearHit_term += np.abs(num)

        nearMiss_term = np.zeros((len(list(set(labels))), n_features))
        for index, label in enumerate(nearMiss.keys()):  # 遍历所有猜错近邻点
            for x in nearMiss[label]:  # 遍历错近邻点点中类别为label的所有点
                nearMiss_term[index] += np.abs(np.power(self_features - x, 2))
                # 猜错近邻 要乘以该类样本在数据集中所占的概率
                # 除以len(nearMiss[label]是为了数据更公平, 消除该类出现不够k个邻近点的误差
            weight += (labels.count(label)/len(labels))*(nearMiss_term[index]/len(nearMiss[label]))
        # 除以len(nearHit)是为了数据更公平, 消除该类出现不够k个邻近点的误差
        weight -= nearHit_term/len(nearHit)

    return weight / iter_ratio  # 最后要除以轮次数目


# 向文件中写数据
def writeToFile(features,labels,filename):
    with open(filename, 'w') as file_object:
        for i in range(features.shape[0]):
            for j in range(len(features[i, :])):
                file_object.write(str(features[i, j]) + ',')
            file_object.write(str(labels[i]) + '\n')


# 计算矩阵的总方差
def calMatVar(targetMat):
    m, n = np.shape(targetMat)  # 获取矩阵的行数, 列数
    TotalVar = 0  # 记录矩阵总的方差
    for i in range(n):
        CVet = targetMat[:, i]
        TotalVar += CVet.var() * m  # 乘以样本个数, 放大方差, 防止数据浮点数过小
    return TotalVar


# 加载数据
def loadDataSet(n_samples, n_features, centers):
    '''
    :param n_samples: 样本个数
    :param n_features: 样本特征数
    :param centers: 样本类别个数
    :return: 规范化之后的样本, 样本标记
    '''
    # 获取个样本, 每个样本个特征值, 类别标签为4的数据集
    features, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    features = normalize(X=features, norm='l2', axis=0)  # 数据规范化, l2是指用范数l2
    return features, labels


def readFromFile(fileName):
    features = []
    labels = []
    with open(fileName) as file_object:
        for line in file_object.readlines():
            lineArr = line.strip().split(',')
            lineArr = list(map(float, lineArr))
            features.append(lineArr[:-1])
            labels.extend(lineArr[-1:])
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    #features, labels = loadDataSet(500, 10, 4)  # 获取500个样本, 每个样本10个特征值, 类别标签为4的数据集
    #print(features)
    #print(labels)
    #weight = np.zeros(len(features[0]))  # 创建权重列表
    #run_times = 10
    #for x in range(run_times):  # 运行run_times次
    #    weight += fit(features=features, labels=labels, iter_ratio=1, k=5, norm='2')/run_times  # 最后结果取平均值
    #print(weight)


    features, labels = readFromFile('dataSet.txt')
    fit(features=features, labels=labels, iter_ratio=1, k=5, norm='2')
    weight = np.zeros(len(features[0]))  # 创建权重列表
    run_times = 1  # 设置运行次数
    for x in range(run_times):  # 运行run_times次
        weight += fit(features=features, labels=labels, iter_ratio=1, k=5, norm='2')/run_times  # 最后结果取平均值


    '''
    这样子不科学, 因为样本带标签, 应该比较信息熵
    # 计算方差占比
    weight_dict = dict()
    for i in range(len(weight)):
        weight_dict[i] = weight[i]
    weight_dict = sorted(weight_dict.items(), key=lambda x:x[1], reverse=True)
    print(weight_dict)
    TotalVar = calMatVar(features)
    childMat = []
    for i in range(len(weight_dict)):
        index = weight_dict[i][0]
        childMat.append(features[:,i])
        ChildVar = calMatVar(np.mat(childMat).T)
        print('前%s个最大的权重对应的方差占比%s'%(i+1,ChildVar/TotalVar*100))
    '''



