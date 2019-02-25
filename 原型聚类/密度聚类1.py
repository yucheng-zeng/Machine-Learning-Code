import math
import numpy as np
import pylab as pl


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2))

# 算法模型
def DBSCAN(D, e, Minpts):
    '''
    :param D: 数据集
    :param e: 半径
    :param Minpts: 最小点个数
    :return: 聚类集合C
    '''
    T = set()  # 初始化核心对象集合T
    k = 0  # 聚类个数k
    C = []  # 聚类集合C
    P = set(D)  # 未访问集合P
    for d in D:  # 遍历每一个样本点
        seed = [ i for i in D if dist(d, i) <= e]  # 以当前点d为圆心,获取样本中所有与之距离小于e的点
        if len(seed) >= Minpts:  # 若果满足最小个数条件
            T.add(d)  # 将d点添加到核心对象集合
    # 开始聚类
    while len(T):
        P_old = P  # 记录拓展之前的未分配的点
        o = list(T)[np.random.randint(0, len(T))]  # 随机获取一个核心点
        P = P - set(o)  # 未访问集合中去除该点
        Q = []  # 用于记录当前聚类簇里面的点
        Q.append(o)  # 将一个核心点增加到当前簇
        # 拓展当前聚类簇
        while len(Q):
            q = Q[0]  # 获取核心点
            Nq = [i for i in D if dist(q, i) <= e]  # 以当前点q为圆心,获取样本中所有与之距离小于e的点
            if len(Nq) >= Minpts:
                S = P & set(Nq)  # 获取 未访问集合P与满足条件的集合的交集
                Q += (list(S))  # 增加点到当前聚类簇里面
                P = P - S  # 从未分配集合里面去除这些点, 因为这这些点都已经被分配过来
            Q.remove(q)  # 当前聚类簇里面去除q, 因为这个点已经拓展过了
        k += 1  # 聚类簇个数+1
        Ck = list(P_old - P)  # 两者之差为一个聚类簇里面的点
        T = T - set(Ck)  # 初始化核心对象集合T减去该聚类的点
        C.append(Ck)  # 聚类集合C增加该聚类的点
    return C

# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []    #x坐标列表
        coo_Y = []    #y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='o', color=colValue[i%len(colValue)], label=i)
    pl.legend(loc='upper right')
    pl.show()



if __name__ == '__main__':
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
    dataset = [(float(a[i]), float(a[i + 1])) for i in range(1, len(a) - 1, 3)]  # 获取数据
    print('dataset=%s'%dataset)
    C = DBSCAN(dataset, 0.11, 5)  # 训练
    draw(C)