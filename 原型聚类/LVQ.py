import re
import math
import numpy as np
import pylab as pl
import copy
data = \
    """1,0.697,0.46,Y,
    2,0.774,0.376,Y,
    3,0.634,0.264,Y,
    4,0.608,0.318,Y,
    5,0.556,0.215,Y,
    6,0.403,0.237,Y,
    7,0.481,0.149,Y,
    8,0.437,0.211,Y,
    9,0.666,0.091,N,
    10,0.639,0.161,N,
    11,0.657,0.198,N,
    12,0.593,0.042,N,
    13,0.719,0.103,N"""


# 定义一个西瓜类，四个属性，分别是编号，密度，含糖率，是否好瓜
class watermelon:
    def __init__(self, properties):
        self.number = properties[0]
        self.density = float(properties[1])
        self.sweet = float(properties[2])
        self.good = properties[3]


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# LVQ 学习向量量化算法
def LVQ(dataset, a=0.01, max_iter=1000, threshold= 1e-5):
    '''
    :param dataset: 数据集
    :param a: 学习率
    :param max_iter: 最大迭代次数
    :return:
    '''
    T = list(set(i.good for i in dataset))  # 统计样本一共有多少个分类
    P = [(i.density, i.sweet) for i in np.random.choice(dataset, len(T))]  # 随机产生个数等于类别个数的原型向量
    #print('P=%s'%P)
    #print(P[0][0])
    #print(len(P[0]))
    while max_iter > 0:
        X = np.random.choice(dataset, 1)[0]  # 随机选择一个西瓜类
        distance = []  # 记录距离
        old_P = copy.deepcopy(P)
        for i in range(len(P)):  # 计算当前点距离原型点的距离
            distance.append(np.sqrt((X.density - P[i][0])**2 + (X.sweet - P[i][1])**2))
        #print('distance',distance)
        #print('min(distance)',min(distance))
        #print('index',distance.index(min(distance)))
        index = distance.index(min(distance))  # 选择离原型向量集中最近的一个点
        #print('index=%s'%index)
        t = T[index]  # 获取距离当前点最近点的标签
        if t == X.good:  # 若果两者标签一样
            # 拉近两者的距离
            P[index] = ((1 - a) * P[index][0] + a * X.density, (1 - a) * P[index][1] + a * X.sweet)  # 更新原型点
        else:
            # 若果两者标签不一样, 使两者远离
            P[index] = ((1 + a) * P[index][0] - a * X.density, (1 + a) * P[index][1] - a * X.sweet)  # 更新原型点

        # 计算变化值
        change = np.sqrt((old_P[index][0] - P[index][0]) ** 2 + (old_P[index][1] - P[index][1]) ** 2)
        #print(change)
        if change < threshold:  # 如果变化值小于阀值, 退出循环
            break
        max_iter -= 1
    return P


def train_show(dataset, P):
    C = [[] for i in P]
    for i in dataset:
        C[i.good == 'Y'].append(i)
    return C


# 画图
def draw(C, P):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j].density)
            coo_Y.append(C[i][j].sweet)
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)
    # 展示原型向量
    P_x = []
    P_y = []
    for i in range(len(P)):
        P_x.append(P[i][0])
        P_y.append(P[i][1])
        pl.scatter(P[i][0], P[i][1], marker='o', color=colValue[i % len(colValue)], label="vector")
    pl.show()


if __name__ == '__main__':
    # 数据简单处理
    a = re.split(',', data.strip(" "))
    dataset = []  # dataset:数据集
    for i in range(int(len(a) / 4)):
        temp = tuple(a[i * 4: i * 4 + 4])
        dataset.append(watermelon(temp))  # 增加一个西瓜类到数据集里面

    P = LVQ(dataset, 0.01, 1000)  # 训练LVQ模型
    C = train_show(dataset, P)  #
    draw(C, P)