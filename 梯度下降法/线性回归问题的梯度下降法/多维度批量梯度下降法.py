import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import style

'''
这李采用的是批量梯度下降法
优点：全局最优解；易于并行实现；总体迭代次数不多
缺点：当样本数目很多时，训练过程会很慢，每次迭代需要耗费大量的时间
'''

# 构造数据
def get_data(sample_num=10000):
    """
    拟合函数为
    y = 5*x1 + 7*x2
    """
    x1 = np.linspace(0, 9, sample_num)  # 创建范围为0～9的等差数列
    x2 = np.linspace(4, 13, sample_num)  # 创建范围为4～13的等差数列
    x = np.concatenate(([x1], [x2]), axis=0).T  # 拼接两个列标x1,x2,变为矩阵,然后取转秩
    y = np.dot(x, np.array([5, 7]).T)  # 矩阵相乘得到目标函数
    return x, y


# 梯度下降法
def GD(samples, y, step_size=0.001, max_iter_count=1000):
    """
    :param samples: 样本
    :param y: 结果value
    :param step_size: 每一接迭代的步长
    :param max_iter_count: 最大的迭代次数
    :param batch_size: 随机选取的相对于总样本的大小
    :return:
    """
    # 确定样本数量以及变量的个数初始化theta值
    m, var = samples.shape  # m 为样本个数, var为样本维度
    theta = np.zeros(2)  # 初始化theta,, 生成一个列表
    print(y)
    y = y.flatten()  # 折叠成一维的列表
    # 进入循环内
    loss = 1  # 记录损失函数值
    iter_count = 0  # 纪律迭代次数
    iter_list = []
    loss_list = []
    theta1 = []
    theta2 = []
    epsilon = 1e-3  # 目标函数与拟合函数的距离当的距离小于epsilo时, 退出

    # 当损失精度大于epsilon且迭代此时小于最大迭代次数时，进行
    while loss > epsilon and iter_count < max_iter_count:
        loss = 0
        # 梯度计算
        theta1.append(theta[0])  # 用于绘图
        theta2.append(theta[1])  # 用于绘图
        for i in range(m):
            h = np.dot(theta, samples[i].T)  # 求出h
            # 更新theta的值,需要的参量有：步长，梯度
            for j in range(len(theta)):  # 更新对应的theta
                theta[j] = theta[j] - step_size * (h - y[i]) * samples[i, j]

        # 计算总体的损失精度，等于各个样本损失精度之和
        for i in range(m):
            h = np.dot(theta.T, samples[i])  # 拟合函数
            # 每组样本点损失的精度
            every_loss = (1 / (var * m)) * np.power((h - y[i]), 2)
            loss = loss + every_loss

        print("iter_count: ", iter_count, "the loss:", loss)

        iter_list.append(iter_count) # 记录迭代次数, 用于绘图
        loss_list.append(loss)  # 添加当前损失函数的值金列表, 用于绘图
        iter_count += 1

    plt.plot(iter_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return theta1, theta2, theta, loss_list  #


# 绘制3D图
def painter3D(theta1, theta2, loss):
    style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.gca(projection='3d')
    x, y, z = theta1, theta2, loss
    ax1.plot(x, y, z)
    ax1.set_xlabel("theta1")
    ax1.set_ylabel("theta2")
    ax1.set_zlabel("loss")
    plt.show()

# 对与未知数据集做预测
def predict(x, theta):
    y = np.dot(theta, x.T)
    return y


if __name__ == '__main__':
    samples, y = get_data()
    theta1, theta2, theta, loss_list = GD(samples, y)
    print(theta)  # 会很接近[5, 7]
    painter3D(theta1, theta2, loss_list)
    predict_y = predict(theta, [7, 8])
    print(predict_y)