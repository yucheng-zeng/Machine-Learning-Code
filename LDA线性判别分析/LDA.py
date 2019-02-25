from numpy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('watermelon_3a.csv')  # 读取数据

def calulate_w():
    df1 = df[df.label == 1]  # 获取所有标记为1的样本点
    df2 = df[df.label == 0]  # 获取所有标记为0的样本点
    X1 = df1.values[:, 1:3]  # 获取所有标记为1的样本点的第一维与第二维的数据, 既是x, y
    X0 = df2.values[:, 1:3]  # 获取所有标记为0的样本点的第一维与第二维的数据, 既是x, y
    mean1 = array([mean(X1[:, 0]), mean(X1[:, 1])])  # 计算mu1, 获取样本点标记为1的均值向量
    mean0 = array([mean(X0[:, 0]), mean(X0[:, 1])])  # 计算mu0, 获取样本点标记为0的均值向量
    m1 = shape(X1)[0]  # 获取标记为1的样本点个数
    sw = zeros(shape=(2, 2))  # 初始化类内散度矩阵
    for i in range(m1):
        # 计算Sigma1
        xsmean = mat(X1[i, :] - mean1)
        sw += xsmean.T * xsmean
    m0 = shape(X0)[0]  # 获取标记为0的样本点个数
    for i in range(m0):
        # 计算Sigma0
        xsmean = mat(X0[i, :] - mean0)
        sw += xsmean.T * xsmean

    w = (mean0 - mean1) * (mat(sw).I)  # 计算w
    return w


def plot(w):
    dataMat = array(df[['density', 'ratio_sugar']].values[:, :])
    labelMat = mat(df['label'].values[:]).transpose()

    m = shape(dataMat)[0]  # 获取样本得到个数

    #print('dataMat=%s'%dataMat)
    #print('labelMat=%s'%labelMat)

    xcord1 = []  # 记录标记为1的x
    ycord1 = []  # 记录标记为1的y
    xcord2 = []  # 记录标记为0的x
    ycord2 = []  # 记录标记为0的x
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.sca(ax)
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.title('LDA')

    plt.figure(2)
    ax = plt.subplot(111)
    mat1 = zeros((len(xcord1),2))
    mat2 = zeros((len(xcord2),2))
    mat1[:,0] = xcord1
    mat1[:,1] = ycord1
    mat2[:,0] = xcord2
    mat2[:,1] = ycord2
    result1 = mat1.dot(w.T)
    result2 = mat2.dot(w.T)
    ax.scatter(list(result1[:, 0]), list(zeros((1,len(result1))).T[:, 0]), s=30, c='red', marker='*')
    ax.scatter(list(result2[:, 0]), list(zeros((1,len(result2))).T[:, 0]), s=30, c='green', marker='^')
    '''
    x = arange(-0.2, 0.2, 0.1)  # 这里随便写, 关键是这条直线的斜率不能变
    y = array((-w[0, 0] * x) / w[0, 1])
    '''

    plt.show()


if __name__ == '__main__':
    w = calulate_w()
    print(w)
    plot(w)