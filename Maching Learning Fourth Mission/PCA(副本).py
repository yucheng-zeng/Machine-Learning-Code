import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def create_data(mean, cov, m):
    x = np.random.multivariate_normal(mean, cov, m)
    return x

def normalize(x):
    mean = np.sum(x, axis=1)/int(x.shape[1])
    mean = mean.reshape((3, 1))
    mean = np.repeat(mean, 100, axis=1)
    x = x - mean
    return x

def PCA(x, top=2):
    XXT = np.dot(x, x.T)
    U, sigma, VT = np.linalg.svd(XXT)
    W = U[:, :top]
    x = np.dot(W.T, x)
    return x

# def paint_3D(x1):
#     ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#     ax.scatter(x1[0, :], x1[1, :], x1[2, :], c='y')  # 绘制数据点
#     ax.set_zlabel('Z')  # 坐标轴
#     ax.set_ylabel('Y')
#     ax.set_xlabel('X')

def paint(x1, x):
    plt.subplot(222)
    plt.scatter(x[0, :], x[1, :], c='y')
    ax = plt.subplot(223, projection='3d')
    ax.scatter(x1[0, :], x1[1, :], x1[2, :], c='r')  # 绘制数据点
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


if __name__ == '__main__':
    mean1 = [1, 1, 1]
    cov1 = 4 * np.eye(3)
    cov1[2][2] = 10

    # mean2 = [10, 10 , 10]
    # cov2 = 4* np.eye(3)
    # cov2[2][2] = 10

    x1 = create_data(mean1, cov1, 100).T
    x1 = normalize(x1)

    # x2 = create_data(mean2, cov2, 100)
    # paint_3D(x1)
    x = np.array(PCA(x1))
    paint(x1, x)
