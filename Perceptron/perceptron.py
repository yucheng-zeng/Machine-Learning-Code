from numpy import *
import operator
import os

def creatDataSet():
    group = array([[3,3], [4,3], [1,1]])  #设置训练数据集
    labels = [1, 1, -1]  # 每个训练数据集对应的标签
    return group, labels

def perceptronClassify(trainGroup, trainLabels):
    global w, b  # 定义变量w, b
    isFind = False  # 定义结束条件的标签
    numSamples = trainGroup.shape[0]  # 获取训练样本的个数
    mLength = trainGroup.shape[1]  # 每个训练样本的维度
    w = [0]*mLength  # 初始化w为0, 其维度与训练样本的维度一致
    b = 0  # 初始化b为0, 其维度为1
    while(not isFind):
        for i in range(numSamples):  # 遍历样本
            if cal(trainGroup[i], trainLabels[i])<=0:  # 计算某个样本点，判断其是否是误分类点
                # 如果是误分类点，更新w,b, 跳出循环，重新遍历，直到没有误分类点
                print(w,b)
                update(trainGroup[i], trainLabels[i], 17)  # 依据当前样本点,更新w,b, 0.5为其步长， 即学习率
                break
            elif i==numSamples-1:  # 如果没有误分类点, 退出循环
                print(w, b)
                isFind = True


def cal(row, trainLabel):  # 计算yi(w*xi+b)
    global w, b
    res = 0
    for i in range(len(row)):
        res += row[i]*w[i]
    res += b
    res *= trainLabel
    return res


def update(row, trainLabel, n):  # 更新w, b
    # w = w + n*yi*xi
    # b = b + n*yi
    global w, b
    for i in range(len(row)):
        w[i] =w[i]+row[i]*trainLabel*n
    b += trainLabel


globals, labels = creatDataSet()
perceptronClassify(globals, labels)
print('f(x)=',w,'x',b)