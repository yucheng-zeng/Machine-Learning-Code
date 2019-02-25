import numpy as np
import math
from sklearn import linear_model
import numpy as np


# 马有三种情况：“仍存活”，“已经死亡”，“已经安乐死”
def colicTest():
    frTrain = open('train_ova.txt'); frTest = open('test_ova.txt')
    trainingSet = []
    trainingLabels = []

    # 获取训练集的数据，并将其存放在list中
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]  # 用于存放每一行的数据
        for i in range(21):  # 这里的range(21)是为了循环每一列的值，总共有22列
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    testingSet = [];
    testingLabels = []

    ##获取测试数据
    for line in frTest.readlines():
        currLine1 = line.strip().split('\t')
        lineArr1 = []  # 用于存放每一行的数据
        for i in range(21):  # 这里的range(21)是为了循环每一列的值，总共有22列
            lineArr1.append(float(currLine1[i]))
        testingSet.append(lineArr1)
        testingLabels.append(float(currLine1[21]))
    return np.array(trainingSet), trainingLabels, np.array(testingSet), testingLabels


##################################训练模型######################################
categorylabels = [0.0, 1.0, 2.0]  # 类别标签


def myweight(dataArr, labelMat, categorylabels):
    weight1 = list()
    for i in range(len(categorylabels)):  # 分成三类，生成三个labelMat，判断是是否和给定的类别标签相等，例如将所有的数据和类别一0.0比较，如果相等令其为1，否则为0
        labelMat1 = []
        for j in range(len(labelMat)):  # 把名称变成0或1的数字
            if labelMat[j] == categorylabels[i]:
                labelMat1.append(1)
            else:
                labelMat1.append(0)
        labelMat1 = np.asarray(labelMat1)  # labelMat1为一个列表，每个元素存放着实际标签和categorylabels的对比
        logreg = linear_model.LogisticRegression(C=1e5)
        a = logreg.fit(dataArr, labelMat1)
        weight1.append(list(a.coef_))
    return weight1


####定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + math.exp(-inX))


def testlabel(dataArr, labelArr, weight):  # 输入的数据均为数组
    initial_value = 0
    list_length = len(labelArr)
    h = [initial_value] * list_length

    for j in range(len(labelArr)):
        voteResult = [0, 0, 0]  # 初始化
        for i in range(3):
            h[j] = float(sigmoid(np.dot(dataArr[j], weight[i][0].T)))  # 数组的点乘，得到训练结果
            if (h[j] > 0.5) and (h[j] <= 1):
                voteResult[i] = voteResult[i] + 1 + h[j]  # 由于类别少，为了防止同票，投票数要加上概率值
            elif (h[j] >= 0) and (h[j] <= 0.5):
                voteResult[i] = voteResult[i] - 1 + h[j]
            else:
                print('Properbility wrong!')
        h[j] = voteResult.index(max(voteResult))
    return np.asarray(h)


def error(reallabel, predlabel):  # reallabel,predlabel分别为真实值和预测值
    error = 0.0
    for j in range(len(reallabel)):
        if predlabel[j] != reallabel[j]:
            error = error + 1
    pro = 1 - error / len(reallabel)  # 正确率
    return pro







dataArr, labelMat, testdata, testlabel = colicTest()
weight1 = myweight(dataArr, labelMat, categorylabels)
mydata, mylabel, testdata, testlabel = colicTest()
h = testlabel(mydata, mylabel, weight1)
###求每个分类器的回归系数
weight1[0]  # 标签为0的分类器
weight1[1]  # 标签为1的分类器
weight1[2]  # 标签为2的分类器
error(mylabel, h)





