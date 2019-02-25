# coding=utf-8

import operator
from calEntropy import *
from paintTree import createPlot, retrieveTree

def createDataSet():
    #创建数据集,加u表示按utf-8编码
    dataSet = [
               ['青年', '否', '否', '一般', '拒绝'],
               ['青年', '否', '否', '好', '拒绝'],
               ['青年', '是', '否', '好', '同意'],
               ['青年', '是', '是', '一般', '同意'],
               ['青年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '一般', '拒绝'],
               ['中年', '否', '否', '好', '拒绝'],
               ['中年', '是', '是', '好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['中年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '非常好', '同意'],
               ['老年', '否', '是', '好', '同意'],
               ['老年', '是', '否', '好', '同意'],
               ['老年', '是', '否', '非常好', '同意'],
               ['老年', '否', '否', '一般', '拒绝'],
               ]
    labels = ['年龄', '有工作', '有房子', '信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, labels
# 从文本之中读取数据
def readDataFromFile(filename):
    labels = []
    dataSet = []
    with open(filename) as fileObject:
        title = fileObject.readline().strip()
    labels = title.split('\t')  #
    labels = [item.strip() for item in labels]  # 获取标签
    with open(filename) as fileObject:
        lines = fileObject.readlines()
    for line in lines:
        sublist = line.split('\t')
        sublist = [item.strip() for item in sublist]
        dataSet.append(sublist)  # 将每个实体加入到数据集之中
    del dataSet[0]  # 删除第一行, 即删除标签
    return dataSet, labels

# 返回每个结点出现次数最多的类别,作为其分类名称
def majorClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #降序排序，可以指定reverse = true
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 用信息增益作为选择特征的条件
# 选择最好的划分方式(选取每个特征划分数据集,从中选取信息增益最大的作为最优划分,即选取信息增益的最大的特征）
def chooseBest(dataSet,labels):
    featNum = len(labels) - 1  # 最后一列是分类, 不能作为分类特征
    baseEntropy = calEmpiricalEntropy(dataSet)  # 经验熵
    bestInforGain = 0.0  # 最好的信息增益的数值
    bestFeature = -1  # 表示最好划分特征的下标

    for i in range(featNum):  # 遍历所有特征, 计算每一个特征的信息增益
        inforGain = calInformationGain(dataSet,baseEntropy,i)
        if (inforGain > bestInforGain):  # 判断, 择优
            bestInforGain = inforGain
            bestFeature = i  # 第i个特征是最有利于划分的特征
    return bestFeature  # 返回最佳特征对应的维度


# 创建树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]  # 获取最后一维的分类信息, 存储到列表中
    if classList.count(classList[0]) == len(classList):  # 如果都是所有的实例都是同树一类, 以该类作为结点的标记,返回单结点树
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果数据集的实例只有一个维度(也就是没有特征), 返回单结点树,以出现次数最多的那个类的值作为该点的类标记
        return majorClass(classList)
    bestFeat = chooseBest(dataSet,labels)  # 选择最好的划分特征的下标
    bestFeatLabel = labels[bestFeat]  # 选择最好的划分特征
    if bestFeat != -1:  # 如果最佳特征为-1, 该特征不能作为分类特征, 则说明前面的特征已经用完, 这时可以结束, 返回树
        #print({bestFeatLabel:{}})
        myTree = {bestFeatLabel:{}}  # 以该特征构建结点
        del(labels[bestFeat])  # 将该特征从特征列表之中移除
        featValues = [example[bestFeat] for example in dataSet]  # 从数据集中获取该特征
        uniqueVals = set(featValues)  # 获取该特征所有可能的取值,放到集合之中
        for value in uniqueVals:  # 以该特征可能的取值将数据集划分为若干子集, 构建子结点, 递归调用, 在子结点基础上,构建子树
            subLabels = labels[:]
            subDataSet = splitDataSet(dataSet, bestFeat, value)  # 划分数据集
            myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)  # 在子结点基础上,构建子树
        return myTree

# 决策树的分类函数, 找到目标结点所对应的结点,返回节点的分类标签
def classify(inputTree,featLabels,testVec):  # 传入的数据为dict类型
    firstSides = list(inputTree.keys())[0]  # 第一个结点的特征
    #print(firstSides)
    firstStr = firstSides[0]  # 第一个结点的特征的值
    #print(firstStr)
    secondDict = inputTree[firstStr]  # 建一个字典, 获取第一个结点的子树
    #print(secondDict)
    featIndex = featLabels.index(firstStr)  #找到在label中特征firstStr的对应的下标
    #print(featIndex)
    for key in secondDict.keys():  # 获取该结点对应的分类条件, Eg：是 或者 否
        #print(key)
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:  # 判断一个变量是否为dict, 等价于判断是否还存在子树, 若果是，继续向下寻找
                classLabel = classify(secondDict[key],featLabels,testVec)
                #print(classLabel)
            else:
                classLabel = secondDict[key]  # 向下没有子树了, 返回当前结点所对应的标签
                #print(classLabel)
    return classLabel   #比较测试数据中的值和树上的值，最后得到节点


#dataSet, labels = createDataSet()
#labelsCopy = labels[:]  # 将labelcopy一份, 因为创建树过程之中, label会被改变, 也省空间
#mytree = createTree(dataSet,labels)
#labels = labelsCopy[:]
#print(mytree)
#print(labels)
#test = ['青年', '否', '否', '一般']
#print(classify(mytree,labels,test))
#createPlot(mytree)



dataSet, labels = readDataFromFile('CARTDataSet(回归树).txt')
labelsCopy = labels[:]
#featIndex = labelsCopy.index('temperature')
#print(featIndex)
mytree = createTree(dataSet, labels)
labels = labelsCopy[:]
print(mytree)
createPlot(mytree)
