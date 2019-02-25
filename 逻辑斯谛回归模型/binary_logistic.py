'''书上没有写出完整的参数估计算法，但给出了其对数似然函数，
经过简单的证明可以得出该函数是单调上升，且其极限为0
因此，我们可以将-L(w)作为损失函数，用随机梯度下降的方法求解'''

import time
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression(object):

    def __init__(self):
        self.learning_step = 0.00001  # 定义学习的步长
        self.max_iteration = 5000  # 定义最大迭代次数

    # 预测某一个样本点的标签取值
    def perdict(self, x):
        dimension = len(self.w)  # 获取每个样本点(即w和x)的维度
        wx = sum([self.w[j]*x[j] for j in range(dimension)])  # 每个维度上的x和w对应相乘,求和,得到wx
        exp_wx = math.exp(wx)

        predict1 = exp_wx / (1 + exp_wx)  # 计算该样本预测为1的概率
        predict0 = 1 / (1 + exp_wx)  # 计算该样本预测为1的概率

        # 返回预测值
        if predict1 > predict0:
            return 1
        else:
            return 0

    # 利用训练数据即训练模型
    def train(self, feature, labels):
        self.w = [0.0] * (len(feature[0])+1)  # 初始化w, 这里的w进行了拓展, 维度为样本实例的维度+1

        correct_count = 0  # 记录样本连续训练正确的次数
        time = 0  # 记录当前迭代次数

        while time < self.max_iteration:  # 若果当前迭代次数小于最大迭代次数, 继续进行迭代
            index = random.randint(0, len(labels)-1)  # 随机抽取一个样本点
            print('index=%s'%index)
            x = list(feature[index])  # 获取数据集中一个样本实例, 存储在x中
            x.append(1.0)  # 对x进行拓展
            y = labels[index]  # 获取该样本的标签

            if y == self.perdict(x):  # 判断预测样本的标签是否与样本的原本标签相等
                correct_count += 1  # 正确次数+1
                if correct_count > self.max_iteration:  # 若果连续预测正确, 说明已经求出较优的w,则可以退出循环
                    break
                continue

            # 预测错误
            time += 1  # 训练次数加1
            correct_count = 0  # 预测错误, 重置为0

            wx = sum([self.w[j]*x[j] for j in range(len(self.w))])  # 获取当前的wx,用与计算梯度
            exp_wx = math.exp(wx)  # 获取当前的exp(wx)用与计算梯度

            for j in range(len(self.w)):
                gradient = (-y*x[j]+float(x[j]*exp_wx)/float(1+exp_wx))  # 计算当前维度为i的梯度
                self.w[j] -= self.learning_step*gradient  # 更新第i维的w

    # 利用已经训练好的模型预测测试数据集的标签
    def predict_dataSet(self, features):
        labels = []  # 新建一个列表,用与储存预测的标签
        for feature in features:  # 遍历每一个测试数据集
            x = list(feature)  # 获取一个样本实例数据
            x.append(1)  # 对x进行拓展
            labels.append(self.perdict(x))  # 预测样本的取值
        return labels

if __name__=="__main__":
    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('data/train_binary.csv', header=0)  # 从数据集里面读取数据, header=0说明第0行作为列名
    data = raw_data.values  # 获取数据

    imgs = data[0::, 1::]  # 读取所有样本, 从第0行到最后一行, 从第一列到最后一列
    labels = data[::, 0]  # 读取所有样本的标签即第0列的数据

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')


    print('Start training')
    lr = LogisticRegression()
    lr.train(train_features, train_labels)  # 训练
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    test_predict = lr.predict_dataSet(test_features)  # 开始预测
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    score = accuracy_score(test_labels, test_predict)  # 判断两个列表的向相似程度, 判断预测的准确率
    print("The accruacy socre is ", score)








