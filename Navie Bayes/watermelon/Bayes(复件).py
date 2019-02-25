import operator
import pandas as pd
from numpy import *

'''
该朴素贝叶斯分类器可以处理连续型数据以及离散型数据
'''

def loadDataSet(filename):
    dataSet = []
    df = pd.read_csv(filename)  # 读取数据
    dataSet = df.values[:, 1:-1]  # 获取数据集的所有样本
    labels = (df.values[:,-1:])[:,0].tolist()  # 获取数据集的标记
    #print(labels)
    #print(dataSet)
    return dataSet, labels

# 计算先验概率
def prior_probability(labels):
    py = {}  # 用于记录先验概率
    labelsSet = set(labels)
    for i in labelsSet:
        py[i] = (labels.count(i)+1)/(len(labels)+len(set(labels)))  # 拉普拉斯修正
    print('py=%s'%py)
    return py

# 计算离散型数据条件概率
def conditional_probability(xj, value, labels):
    xcount = {}  # 记录出现次数
    labelsSet = set(labels)  # 记录标签集合
    for i in range(len(labels)):
        for label in labelsSet:
            if labels[i] == label and xj[i] == value:
                xcount[label] = xcount.get(label, 0) + 1  # 出现次数加一

    pxy = {}  # 记录条件概率
    for label in labelsSet:
        pxy[label] = (float(xcount[label])+1)/(labels.count(label)+len(set(xj)))  # 计算条件概率, 拉普拉斯修正
    print('pxy=%s'%pxy)
    return pxy


# 计算连续型数据的概率密度分布
def conditional_probability_continuous(xj, value, labels):
    labelsSet = set(labels)  # 记录标签集合
    dataSet = {}  # 记录在当前特征下,　标签为不同类别的数据
    #print(xj)
    # 寻找标签为不同类别的数据
    for label in labelsSet:
        dataSet[label] = []  # 初始化
        for index in range(len(labels)):
            if labels[index] == label:
                dataSet[label].append(xj[index])  # 添加数据到键为label的列表
    attribute_mean = {}  # 储存平均值
    attribute_std = {}  # 储存方差
    for label in labelsSet:  #
        attribute_mean[label] = mean(dataSet[label])  # 计算标签为label的数据平均值
        attribute_std[label] = std(dataSet[label])  # 计算标签为label的数据方差
    #print(attribute_mean)
    #print(attribute_std)
    pxy = {}  # 记录条件概率
    for label in labelsSet:
        # 计算条件概率密度分布
        pxy[label] = (1/(sqrt(2*pi)*attribute_std[label]))*exp(-((value-attribute_mean[label])**2)/(2*attribute_std[label]**2))
    print('pxy=%s'%pxy)
    return pxy
# 判断是否为连续型数据
def isDigit(X):
    try:
        X = float(X)
        return True
    except:
        return False


# 朴素贝叶斯分类器
def classify(X, inX, labels):
    pxy = {}  # 用于计算条件概率
    n = X.shape[1]  # 获取样本的特征数目
    for i in range(n):  # 计算每一个特征的条件概率
        if not isDigit(inX[i]):  # 若果是离散型数据, 计算条件概率
            pxy[i] = conditional_probability(X[:, i],inX[i],labels)  # 计算离散型的条件概率
        else:  # 若果是连续型数据, 计算概率密度函数
            pxy[i] = conditional_probability_continuous(X[:, i],inX[i],labels)  # 计算连续型数据的概率密度分布
    py = prior_probability(labels)  # 计算先验概率
    pVec = {}  # 用于记录贝叶斯分类器的结果
    labelsSet = set(labels)  # 记录标签集合
    # 计算贝叶斯分类器的结果
    for label in labelsSet:  # 遍历每一个标签的取值
        pVec[label] = py.get(label)
        for i in range(n):
            pVec[label] = pVec[label] * pxy[i].get(label)
    print(pVec)
    sortedClassCount = sorted(pVec.items(), key=operator.itemgetter(1), reverse=True)  # 按结果从大到小排序
    return sortedClassCount[0][0]  # 取概率最大的标签


if __name__=='__main__':
    dataSet, labels = loadDataSet('bayes.csv')
    inX = ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460]
    result = classify(dataSet, inX, labels)
    print(result)
