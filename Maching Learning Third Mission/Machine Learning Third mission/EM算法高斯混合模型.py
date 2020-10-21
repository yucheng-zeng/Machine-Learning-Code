import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import csv


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  # 将数据拆开返回一个列表
        fltLine = list((map(float, curLine)))  # 将列表里面的数字转化为浮点类型
        dataMat.append(fltLine)  # 添加到列表之中
    return dataMat  # 返回一个列表


# 生成随机数据，生成一个高斯模型
def createDataSet(means, covs, N):
    dataSet = np.random.multivariate_normal(means[0], covs[0], N[0])
    for i in range(1, len(N)):
        x = np.random.multivariate_normal(means[i], covs[i], N[i])
        dataSet = np.vstack((dataSet, x))
    return dataSet

class trainEm(object):
    def __init__(self, dataSet, k, iter_num, sigma, epsilon=1e-6):
        '''
        :param dataSet: 数据集
        :param k: 中心簇个数
        :param iter_num: 最大迭代次数
        :param epsilon: 阀值
        '''
        self.dataSet = dataSet
        self.k = k
        self.item_num = iter_num
        self.epsilon = epsilon
        self.gama = np.zeros((len(dataSet),k))
        self.mu = np.mat(np.random.random((k, dataSet.shape[1])))  #
        self.sigma = sigma*np.eye(dataSet.shape[1])
        for i in range(1, k):
            self.sigma = np.vstack((self.sigma, sigma*np.eye(dataSet.shape[1])))
        self.sigma = self.sigma.reshape((k, dataSet.shape[1], dataSet.shape[1]))
        self.alpha = [1/k]*k  # 开始时, k个模型处于同等地位, 权重一样

    def train(self):
        for i in range(0, iter_num):
            old_mu = copy.deepcopy(self.mu)
            self.e_step()
            self.m_step()
            diff = self.cal_diff(old_mu, self.mu)
            if diff <= self.epsilon:
                #print('iter=',i, 'mu_diff=',diff)
                return
        return

    def e_step(self):
        for j in range(0, len(dataSet)):
            for k in range(0, self.k):
                mole = self.alpha[k]*self.cal_density(self.dataSet[j],self.mu[k], self.sigma[k])
                deno = 0
                for ik in range(0, self.k):
                    deno += self.alpha[ik]*self.cal_density(self.dataSet[j], self.mu[ik], self.sigma[ik])
                self.gama[j, k] = mole/deno

    def m_step(self):
        # update parameter
        for k in range(0, self.k):
            mu_mole = 0
            mu_deno = 0
            sigma_mole = 0
            sigma_deno = 0
            alpha_mole = 0
            alpha_deno = 0
            for j in range(0, len(self.dataSet)):
                mu_mole += self.gama[j, k]*self.dataSet[j]
                mu_deno += self.gama[j, k]
                #sigma_mole += np.dot(self.gama[j, k], (self.dataSet[j]-self.mu[k])**2)
            sigma_deno = mu_deno
            alpha_mole = mu_deno
            alpha_deno = len(self.dataSet)
            self.mu[k] = mu_mole/mu_deno
            # self.sigma[k] = sigma_mole/(sigma_deno+1e-10)
            self.alpha[k] = alpha_mole/alpha_deno

    def cal_diff(self, new_mu, old_mu):
        return np.linalg.norm(new_mu-old_mu)

    def cal_density(self, x, mu, sigma):
        inv = np.linalg.inv(sigma)
        x_mu = np.mat(x-mu)
        temp = np.dot(np.dot((x_mu), inv), (x_mu).T)[0,0]
        mole = np.exp(temp/-2.0)
        deno = np.sqrt(((2*np.pi)**self.dataSet.shape[1])*np.linalg.det(sigma))
        return mole/deno

def paintView(dataSet, mu):
    plt.scatter(list(dataSet[:, 0]), list(dataSet[:, 1]), c='red', marker='o')
    plt.scatter(list(mu[:, 0]), list(mu[:, 1]), c='green', s=50, marker='x')
    plt.show()

# 计算欧氏距离
def distEclud(vecA, vecB):
    return np.sqrt(sum((vecA - vecB)**2))  # 计算欧氏距离

def cal_loss(dataSet, centroids):
    cluster = {}
    for i in range(0, centroids.shape[0]):
        cluster[i] = []
    loss = 0
    for i in range(0, dataSet.shape[0]):
        minDist = np.inf
        minIndex = -1
        for k in range(0, centroids.shape[0]):
            vecC = []
            vecC.append(centroids[0, 0])
            vecC.append(centroids[0, 1])
            dist = distEclud(dataSet[i, :], vecC)
            if dist < minDist:
                minIndex = k
                minDist = dist
        cluster[minIndex].append(dataSet[i:])
        loss += minDist
    return loss

if __name__ == '__main__':

    mean1 = [1, 5]
    cov1 = 0.1 * np.eye(2)

    mean2 = [1, 1]
    cov2 = 0.1 * np.eye(2)

    mean3 = [5, 1]
    cov3 = 0.1 * np.eye(2)

    mean4 = [5, 5]
    cov4 = 0.1 * np.eye(2)

    N = [40, 40, 40, 40]  # 每类样本各生成多少个
    means = np.vstack((np.vstack((np.vstack((mean1, mean2)),mean3)), mean4))
    covs = np.vstack((np.vstack((np.vstack((cov1, cov2)),cov3)), cov4)).reshape((4, 2, 2))
    K = 4  # 中心簇个数
    iter_num = 100  # 迭代次数

    dataSet = createDataSet(means, covs, N)

    # for i in range(3, 10):
    #     train = trainEm(dataSet, k=i, iter_num=iter_num, sigma=1, epsilon=1e-5)  # 训练模型
    #     train.train()
    #     loss = cal_loss(dataSet, train.mu)
    #     print('K=',i,'Loss=',loss)


    trainEm = trainEm(dataSet, K, iter_num=iter_num, sigma=30, epsilon=1e-5)  # 训练模型
    trainEm.train()
    cal_loss(dataSet, trainEm.mu)
    paintView(dataSet, trainEm.mu)


