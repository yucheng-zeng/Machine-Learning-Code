from math import *
# 计算训练数据集合的经验熵
def calEmpiricalEntropy(dataSet):

    numEntries = len(dataSet)  # 实例个数
    labelCounts = {}  # 用于保存标签,及其出现次数
    for featVec in dataSet:  # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]  # 获取每个实例的最后一维, 即同意与否
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 初始化
        labelCounts[currentLabel] += 1  # 记录标签出现次数
    empiricalEnt = 0.0  # 经验熵
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 每个标签占样本的总数的概率
        empiricalEnt += -prob*log(prob, 2)  # 计算经验熵
    return empiricalEnt

def calConditionalEntropy(dataSet, i, featList, uniqueVals):
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)  # 获取划分之后的数据集的子集
        prob = len(subDataSet) / float(len(dataSet))  # 极大似然估计概率
        ce += prob * calEmpiricalEntropy(subDataSet)  # ∑pH(Y|X=xi) 条件熵的计算
    return ce


# 划分数据集（以指定特征将数据进行划分）
def splitDataSet(dataSet,feature,value):  # 入待划分的数据集、划分数据集的特征以及需要返回子集所对应的特征的值
    newDataSet = []
    for featVec in dataSet:  # 从数据集中获取每一个实例
        if featVec[feature] == value:  # 如果实例的特征与参数特征相同, 提出这个样本点
            # 要将这个特征从提取出来的样本中剔除
            reducedFeatVec = featVec[:feature]
            reducedFeatVec.extend(featVec[feature + 1:])
            newDataSet.append(reducedFeatVec)  # 将这个实例纳入新的数据集
    return newDataSet

# 计算信息增益
def calInformationGain(dataSet, baseEntropy, i):
    featList = [example[i] for example in dataSet]  # 第i维特征列表
    uniqueVals = set(featList)  # 转换成集合
    newEntropy = calConditionalEntropy(dataSet, i, featList, uniqueVals)  # 计算概特征的条件熵
    infoGain = baseEntropy - newEntropy  # 信息增益，就是熵的减少，也就是不确定性的减少
    return infoGain

# 计算信息增益比
def calInformationGainRate(dataSet, baseEntropy, i):
    return calInformationGain(dataSet, baseEntropy, i) / baseEntropy




