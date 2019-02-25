import pandas as pd
import numpy as np
import time
import math
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np


def loadData(fileName):
    trainset = []  # 训练数据集
    w = []  # 参数向量
    labels = set([])  # 标签
    features = defaultdict(int)  # 用于获得(标签，特征)键值对
    for line in open(fileName):
        fields = line.strip().split()  # 分割数据
        # at least two columns
        if len(fields) < 2:  # 过滤数据
            continue  # 只有标签没用
        label = fields[0]  # 第一列为标签
        labels.add(label)  # 获取label

        for f in set(fields[1:]):  # 对于每一个特征
            features[(label, f)] += 1  # 每提取一个（标签，特征）对，就自加1，统计该特征-标签对出现了多少次

        trainset.append(fields)  # 添加到训练集合
        w = [0.0] * len(features)  # 初始化权重
        lastw = w  # 迭代前的w
    return trainset, features, labels, w, lastw

def initP(trainset,features,labels,w,lastw):
    # 获得M
    M = max([len(feature[1:]) for feature in trainset])  # M值为最大特征数目
    size = len(trainset)  # 获取样本点个数
    Ep_ = [0.0] * len(features)  # 初始化期望值,维度等于每个样本的（标签，特征）个数

    # 获得联合概率期望
    for i, feat in enumerate(features):  # i表示(标签，特征）对下标, feat表示(标签，特征）对
        Ep_[i] += features[feat] / (1.0 * size)  # 获得联合概率期望
        # 更改键值对为（label-feature）-->id,每一个(标签，特征）对对应一个下标
        features[feat] = i  #
    # 准备好权重
    w = [0.0] * len(features) # 初始化w, 维度等于(标签，特征）
    lastw = w  # 迭代前的w
    return trainset,features,labels,w,lastw, M, Ep_,size

def train(trainset,features,labels,w,lastw,max_iter=1000):  # 设置最大步数

    trainset, features, labels, w, lastw, M, Ep_, size = initP(trainset,features,labels,w,lastw)  # 主要计算M以及联合分布在f上的期望

    # Ep_是特征函数f(x,y)关于经验分布P_(x,y)的期望值

    # 下面计算条件分布及其期望，正式开始训练
    for i in range(max_iter):  # 计算条件分布在特诊函数上的期望
        Ep = EP(trainset,features,labels,w,size)  # Ep 是模型P(X|Y)与经验分布P(X)的期望值
        lastw = w[:] # 保存迭代前的w
        print(w)
        for i, u in enumerate(w):
            theta = (1.0 / M) * np.log(Ep_[i] / Ep[i])  # 计算出theta(i)
            w[i] = w[i] + (theta)  # 更新w[i]
        if convergence(w,lastw):  # 判断是否要退出循环
            break

#　计算模型P(X|Y)与经验分布P(X)的期望值
def EP(trainset,features,labels,w,size):
    # 计算p（y|x）
    ep = [0.0] * len(features)  # 初始化模型P(X|Y)与经验分布P(X)的期望值, 维度等于(标签，特征)对的个数
    for record in trainset:  # 遍历数据集
        onefeatures = record[1:]  # 回去每一个样本的特征
        # 计算P(X|Y)
        prob = calPyx(onefeatures,labels,features,w)
        for f in onefeatures:  # 特征一个个来
            for pyx, label in prob:  # 获得条件概率与标签
                if (label, f) in features:  # 获取对应的（标签，特征）对存在
                    id = features[(label, f)]  # 获取id
                    ep[id] += (1.0 / size) * pyx  # 计算相应的期望
    return ep

# 获得最终单一样本每个特征的p（y|x）
def calPyx(onefeatures,labels,features, w):
    # 传的onefeatures是单个样本的
    '''
    for label in labels:  # 便利每一个标签获取
        SumP = calSumP(onefeatures, label)  # 计算分母
        wlpair = [(SumP, label)]  # 遍历标签,构建(标签，特征)对,计算期望值
    '''
    wlpair = [(calSumP(onefeatures, label, features, w), label) for label in labels]  # 遍历标签,构建(标签，特征)对,计算期望值
    Z = sum([w for w, l in wlpair])  # 计算w的和
    prob = [(w / Z, l) for w, l in wlpair]
    return prob

#　计算分母
def calSumP(onefeatures, label, features, w):
    '''书本85页的分母Zw(x)'''
    sumP = 0.0
    #print('2%s'%onefeatures)
    # 对于这单个样本的feature来说，不存在于feature集合中的f=0所以要把存在的找出来计算
    for showedF in onefeatures:  # 获得单个样本的所有(标签，特征)键值对
        if (label, showedF) in features:
            index = features[(label, showedF)]  # 获取每一个标签对在w中的下标
            sumP += w[index]  # 获取每一个(标签，特征)键值对的w
    return np.exp(sumP)  # 返回sumP

# 判断是否结束
def convergence(w,lastw):
    for i in range(len(w)):
        if abs(w[i] - lastw[i]) >= 0.001:  # 如果w与lastw的一个维度
            return False
    return True

# 预测
def predict(input,labels,features, w):
    features = input.strip().split()
    prob = calPyx(features,labels,features,w)
    prob.sort(reverse=True)
    return prob

trainset, features, labels, w, lastw = loadData('gameLocation.dat')
train(trainset, features, labels, w, lastw)
print(predict('Sunny',labels,features,w))
