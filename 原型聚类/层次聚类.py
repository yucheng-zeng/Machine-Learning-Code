import math
import pylab as pl


# 加载数据, 预处理数据
def loadData(filename):
    dataSet = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 将数据映射为浮点型数据
        dataSet.append(fltLine)
    return dataSet


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))


# 最小距离
def dist_min(Ci, Cj):
    return min(dist(i, j) for i in Ci for j in Cj)


# 最大距离
def dist_max(Ci, Cj):
    return max(dist(i, j) for i in Ci for j in Cj)


# 平均距离
def dist_avg(Ci, Cj):
    return sum(dist(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

# 找到距离最小的两个聚类簇的下标, 以及最小距离
def find_Min(M):
    min = 1000
    x = 0; y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return x, y, min


# 计算聚类之间的距离
def cluster_dist(C, dist):
    M = []  # 记录聚类之间的距离
    for i in C:  # 遍历所有的聚类
        Mi = []
        for j in C:
            Mi.append(dist(i, j))  # 计算当前聚类i与聚类j之间的距离
        M.append(Mi)
    return M


# AGNES 算法
def AGNES(dataset, dist, k):
    '''
    :param dataset: 数据集
    :param dist: 距离计算方式
    :param k: 聚类簇个数
    :return: 聚类簇
    '''

    C = []  # 初始化聚类C
    M = []  # 记录聚类之间的距离
    for i in dataset:  # 遍历所有样本
        Ci = []
        Ci.append(i)
        C.append(Ci)
    M = cluster_dist(C, dist)  # 计算聚类之间的距离
    q = len(C)  # 获取聚类簇的个数
    # 合并更新
    while q > k:
        x, y, min = find_Min(M)  # 找到距离最小的两个聚类簇的下标, 以及最小距离
        C[x].extend(C[y])  # 合并两个簇
        C.remove(C[y])  # 删除被合并的簇C[y]
        M = cluster_dist(C, dist)  # 重新计算合并之后的两个簇之间的距离
        q -= 1  # 聚类簇的个数减一
    return C  # 返回聚类簇


#画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='o', color=colValue[i%len(colValue)], label='cluster'+str(i+1))

    pl.legend(loc='upper right')
    pl.show()

if __name__ == '__main__':
    data = loadData('testSet.txt')
    dataset = [(float(item[0]), float(item[1])) for item in data]
    #print(dataset)
    '''
    # 数据集：每三个是一组分别是西瓜的编号，密度，含糖量
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""

    # 数据处理 dataset是30个样本（密度，含糖量）的列表
    a = data.split(',')
    dataset = [(float(a[i]), float(a[i + 1])) for i in range(1, len(a) - 1, 3)]
    print(dataset)
    '''
    C = AGNES(dataset, dist_avg, 4)
    draw(C)