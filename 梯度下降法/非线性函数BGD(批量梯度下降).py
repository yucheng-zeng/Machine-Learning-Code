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

def createDataSet():
    '''
    拟合函数
    y = 5*x**2 + 6
    '''
    # 原理一样,采用便梁代换的方法使u=x**2,这是方程又回到线性回归函数y=5*u+6
    x1 = np.linspace(-5, 5, 100)
    x2 = np.full(len(x1),1.0)
    x = np.concatenate(([x1**2],[x2]),axis=0).T
    y = np.dot(x,np.array([5,6]).T)
    return x, y, x1  # 这里的x以返回用于计算绘图

# 梯度下降法
def GD(samples, y, step_size=0.0001, max_iter_count=5000):
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
    theta = np.zeros(2)  # 初始化theta, 生成一个列表
    y = y.flatten()  # 折叠成一维的列表
    # 进入循环内
    loss = 1  # 记录损失函数值
    iter_count = 0  # 纪录迭代次数,用于绘图
    iter_list = []  # 记录走过的步数,用于绘图
    loss_list = []  # 纪录损失函数的取值,用于绘图
    theta1 = []  # 记录theta1的取值,用于绘图
    theta2 = []  # 记录theta1的取值,用于绘图
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

        iter_list.append(iter_count) #
        loss_list.append(loss)  # 添加当前损失函数的值金列表, 用于绘图
        iter_count += 1
    print(theta)
    plt.plot(iter_list, loss_list)
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.show()
    return theta1, theta2, theta, loss_list  #

# 绘制3D图
def painter3D(theta1, theta2, loss):
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
    x = [x**2,1]
    y = np.dot(x,np.array([theta[0],theta[1]]))
    return y


if __name__ == '__main__':
    samples, y, x= createDataSet()
    theta1,theta2,theta,loss_list = GD(samples, y)
    plt.plot(x, y, 'g.')
    plt.plot(x, theta[0]*x**2 + theta[1], 'r')
    plt.show()
    painter3D(theta1,theta2,loss_list)
    predict_y = predict(7,theta)
    print(predict_y)