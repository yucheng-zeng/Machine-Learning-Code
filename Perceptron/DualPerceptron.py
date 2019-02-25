from numpy import *
import operator
import os

def creatDataSet():
    group = array([[3,3], [4,3], [1,1]])  #设置训练数据集
    labels = [1, 1, -1]  # 每个训练数据集对应的标签
    return group, labels

def dualPerceptronClassify(trainGroup, trainLabels):
    global a, b  # 定义变量a, b
    isFind = False  # 定义结束条件的标签
    numSamples = trainGroup.shape[0]  # 获取训练样本的个数
    a = [0]*numSamples  # 初始化a为0, 其维度与训练样本个数一致
    b = 0  # 初始化b为0, 其维度为1
    gMatrix = cal_gram(trainGroup)  # 获得gram矩阵
    while(not isFind):
        for i in range(numSamples):  # 遍历样本
            if cal(gMatrix, trainLabels, i)<=0:  # 计算某个样本点，判断其是否是误分类点
                # 如果是误分类点，更新a, b, 跳出循环，重新遍历，直到没有误分类点
                cal_wb(trainGroup, trainLabels)
                update(i, trainLabels[i],1)  # 依据当前样本点,更新w,b, 0.5为其步长， 即学习率
                break
            elif i==numSamples-1:  # 如果没有误分类点, 退出循环
                cal_wb(trainGroup, trainLabels)  # 计算出最终的w,b
                isFind = True


def cal(gMatrix, trainLabels, key):  # 计算yi( (连加(j=1——>N)aj*yj*xj*xi) +b),判断其是否是误分类
    global a, b
    res = 0
    for j in range(len(trainLabels)):
        res += a[j]*trainLabels[j]*gMatrix[j][key]
    res = (res + b)*trainLabels[key]
    return res


def update(i, trainLabel, n):  # 更新a, b, n为步长，即为学习率
    global a, b
    a[i] += n  # ai = ai + n
    b += trainLabel  # bi = bi +n*yi

def cal_gram(trainGroup):  # 计算Gram矩阵(N*N), 即计算训练样本内积
    mLength = trainGroup.shape[0]  # 获取训练样本的个数
    gMatrix = zeros((mLength,mLength))  # 初始化Gram矩阵,创建一个(N*N)的零矩阵
    for i in range(mLength):
        for j in range(mLength):
            gMatrix[i][j] = dot(trainGroup[i],trainGroup[j])  # 矩阵点乘, dot的两个参数为两个行/列向量
    return gMatrix

def cal_wb(trainGroup, trainLables):  # 计算出依据a,b, 计算出w,b,这里的两个b可以认为是不一样
    global a, b
    w = [0]*(trainGroup.shape[1]) # w的维度与样本点的一样
    h = 0  # 维度为1
    for i in range(len(trainLables)):
        h += a[i]*trainLables[i]  # b = (连加(i=1——>N)ai*yi)
        w += a[i]*trainLables[i]*trainGroup[i]  # w = (连加(i=1——>N)ai*yi*xi)
    print(w, h)




globals, labels = creatDataSet()
dualPerceptronClassify(globals, labels)