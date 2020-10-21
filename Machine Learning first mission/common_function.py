import numpy as np
import matplotlib.pyplot as plt

class config:
    def __init__(self, x0, x1, n, k, alpha, loop_max, epsilon, theta):
        '''
        :param x0: (X0,X1)指定数据范围
        :param x1: (X0,X1)指定数据范围
        :param n: n指定数据个数
        :param k: K指定多项式复杂度
        :param alpha: 学习率
        :param loop_max: 最大迭代次数
        :param epsilon: 阀值
        :param theta: 正则化系数
        '''
        self.x0 = x0
        self.x1 = x1
        self.n = n
        self.k = k
        self.alpha = alpha
        self.loop_max = loop_max
        self.epsilon = epsilon
        self.theta = theta

# 创造数据
def create_data(x0=-5, x1=5, n=20):
    x = np.arange(x0, x1, (x1 - x0) / n)
    y = np.sin(x)
    y1 = y + np.random.rand(len(x)) * 0.2  # 噪声
    return x, y1

# 构建矩阵
def create_mat(x,y,k):
    matX = []
    #构建矩阵X
    for xi in x:
        temp = 1
        col = []
        for i in range(k):
            col.append(temp)
            temp = temp*xi
        matX.append(col)
    matX = np.array(matX)
    #构建矩阵Y
    matY = np.array(y)
    return matX, matY

# 评价
def RMSE(matX,matY,matA):
    result = matY-matX.dot(matA)
    result = result * result
    loss = np.mean(result)
    return loss

# 画图
def show(x,y,matX, matA):
    plt.plot(x, y, color='b', linestyle='', marker='.')
    plt.plot(x, matX.dot(matA), color='r', linestyle='-', marker='')
    plt.show()
