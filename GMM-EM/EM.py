import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import random
import csv

def loadDataSet(filename):
    dataSet = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dataSet.append(row)
    return dataSet

class trainEm(object):
    def __init__(self, dataSet, k, iter_num, sigma, epsilon=1e-6):
        '''
        :param dataSet: 数据集
        :param k: 中心簇个数
        :param iter_num: 最大迭代次数
        :param sigma: 初始化方差矩阵
        :param epsilon: 阀值
        '''
        self.dataSet = dataSet
        self.k = k
        self.item_num = iter_num
        self.epsilon = epsilon
        self.gama = np.zeros((len(dataSet),k))   #
        self.mu = np.mat(random.sample(list(dataSet), k))
        self.sigma = sigma*np.eye(dataSet.shape[1])
        for i in range(1, k):
            self.sigma = np.vstack((self.sigma, sigma*np.eye(dataSet.shape[1])))
        self.sigma = self.sigma.reshape((k, dataSet.shape[1], dataSet.shape[1]))
        self.alpha = [1/k]*k  # 开始时, k个模型处于同等地位, 权重一样

    def train(self):
        for i in range(0, self.item_num):
            print(i)
            old_mu = copy.deepcopy(self.mu)
            self.e_step()
            self.m_step()
            if np.linalg.norm(self.mu-old_mu) <= self.epsilon:
                return
        return

    def e_step(self):
        for j in range(0, len(self.dataSet)):
            deno = 0
            for ik in range(0, self.k):
                deno += self.alpha[ik] * cal_density(self.dataSet[j], self.mu[ik], self.sigma[ik])
            for k in range(0, self.k):
                mole = self.alpha[k] * cal_density(self.dataSet[j], self.mu[k], self.sigma[k])
                self.gama[j, k] = float(mole)/float(deno)

    def m_step(self):
        # update parameter
        for k in range(0, self.k):
            mu_mole = 0
            mu_deno = 0
            sigma_mole = 0
            for i in range(0, self.dataSet.shape[0]):
                mu_mole += self.gama[i, k]*self.dataSet[i]
                mu_deno += self.gama[i, k]
                sigma_mole += self.gama[i, k]*(self.dataSet[i]-self.mu[k]).T*(self.dataSet[i]-self.mu[k])
            sigma_deno = mu_deno
            alpha_mole = mu_deno
            alpha_deno = len(self.dataSet)
            self.mu[k] = mu_mole/mu_deno
            self.sigma[k] = sigma_mole/sigma_deno
            self.alpha[k] = alpha_mole/alpha_deno

    def cal_diff(self):
        MLE = 0
        for i in range(0, self.dataSet.shape[0]):
            sum_density = 0
            for k in range(0, self.k):
                sum_density += self.alpha[k]*cal_density(self.dataSet[i],self.mu[k], self.sigma[k])
            MLE += math.log(sum_density)
        return MLE

def cal_density(x, mu, sigma):
    inv = np.linalg.inv(sigma)
    x_mu = np.mat(x - mu)
    temp = np.dot(np.dot((x_mu), inv), (x_mu).T)[0, 0]
    mole = np.exp(temp / -2.0)
    deno = np.sqrt(((2 * np.pi) ** x.shape[0]) * np.linalg.det(sigma))
    return mole / deno

def classify(testData, modelList, labelMap):
    label = []
    for i in range(0, testData.shape[0]):
        dists = []
        for model in modelList:
            sum = 0
            # 计算模型概率密度
            for k in range(0, model.k):
                pro = model.alpha[k]*cal_density(testData[i], model.mu[k], model.sigma[k])
                sum += pro
            dists.append(sum)
        label.append(labelMap[dists.index(max(dists))])  # 找出概率最大的模型，作为标签
    return label

def GMMClassify():
    K = 2  # 中心簇个数
    iter_num = 100  # 迭代次数
    dataSet1 = loadDataSet('./data/Train1.csv')
    dataSet1 = np.array(dataSet1).astype(float)

    dataSet2 = loadDataSet('./data/Train2.csv')
    dataSet2 = np.array(dataSet2).astype(float)

    trainEm1 = trainEm(dataSet1, K, iter_num=iter_num, sigma=2, epsilon=1e-5)  # 训练模型
    trainEm1.train()

    trainEm2 = trainEm(dataSet2, K, iter_num=iter_num, sigma=2, epsilon=1e-5)  # 训练模型
    trainEm2.train()

    testData1 = loadDataSet('./data/test1.csv')
    testData1 = np.array(testData1).astype(float)

    testData2 = loadDataSet('./data/test2.csv')
    testData2 = np.array(testData2).astype(float)

    labelMap = [0, 1]
    modelList = [trainEm1, trainEm2]
    label1 = classify(testData1, modelList, labelMap)
    sum = 0
    for i in range(0, len(label1)):
        if label1[i] == 0:
            sum += 1
    print(sum, float(sum) / len(label1))

    sum = 0
    label2 = classify(testData2, modelList, labelMap)
    for i in range(0, len(label2)):
        if label2[i] == 1:
            sum += 1
    print(sum, float(sum) / len(label2))

def Mnist():
    dataSet = loadDataSet('./data/TrainSamples.csv')
    dataSet = np.array(dataSet).astype(float)
    label = loadDataSet('./data/TrainLabels.csv')
    label = np.array(label).astype(int)

    K = 5
    iter_num = 10  # 迭代次数
    sigma = 50000
    epsilon = 1e-5
    classifiedDataSet = [[], [], [], [], [], [], [], [], [], []]
    modelList = []
    for i in range(0, len(label)):
        temp = label[i, 0]
        classifiedDataSet[temp].append(dataSet[i])
    for i in range(0, 10):
        tmpDataSet = np.array(classifiedDataSet[i]).astype(float)
        modelList.append(trainEm(tmpDataSet, K, iter_num, sigma, epsilon))
        modelList[i].train()

    labelMap = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    testDataSet = loadDataSet('./data/TestSamples.csv')
    testDataSet = np.array(testDataSet).astype(float)
    testLabel = loadDataSet('./data/TestLabels.csv')
    testLabel = np.array(testLabel).astype(int)
    label = classify(testDataSet, modelList, labelMap)

    sum = 0
    for i in range(0, len(label)):
        if label[i] == testLabel[i]:
            sum += 1
    print(sum, sum/len(label))

    return
if __name__ == '__main__':
    Mnist()
