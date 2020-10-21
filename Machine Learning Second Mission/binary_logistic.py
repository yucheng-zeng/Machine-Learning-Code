import time
import math
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle as pkl
import os

class LogisticRegression(object):

    def __init__(self, learning_step=1e-10, max_iteration=1, alpha=0.01):
        '''
        :param learning_step: 学习率
        :param max_iteration: 最大迭代次数
        :param alpha: 惩罚项系数
        '''
        self.learning_step = learning_step  # 定义学习的步长
        self.max_iteration = max_iteration  # 定义最大迭代次数
        self.alpha = alpha
        self.w = []

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
    def train(self, train_features, train_labels, test_features, test_labels):
        self.w = [0.0] * (len(train_features[0])+1)  # 初始化w, 这里的w进行了拓展, 维度为样本实例的维度+1
        time = 0  # 记录当前迭代次数
        while time < self.max_iteration:  # 若果当前迭代次数小于最大迭代次数, 继续进行迭代
            for index in range(0, len(train_labels)):
                x = list(train_features[index])  # 获取数据集中一个样本实例, 存储在x中
                x.append(1.0)  # 对x进行拓展
                y = train_labels[index]  # 获取该样本的标签
                # 计算梯度，并且更新w
                self.cal_gradient(x, y)
                #self.cal_gradient_with_regular(x, y)
            time += 1
        return self.w

    def cal_gradient(self, x, y):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  # 获取当前的wx,用与计算梯度
        exp_wx = math.exp(wx)  # 获取当前的exp(wx)用与计算梯度

        for j in range(len(self.w)):
            gradient = (-y * x[j] + float(x[j] * exp_wx) / float(1 + exp_wx))  # 计算当前维度为i的梯度
            self.w[j] -= self.learning_step * gradient  # 更新第i维的w
        return

    def cal_gradient_with_regular(self, x, y):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])  # 获取当前的wx,用与计算梯度
        exp_wx = math.exp(wx)  # 获取当前的exp(wx)用与计算梯度

        for j in range(len(self.w)):
            gradient = (-y * x[j] + float(x[j] * exp_wx) / float(1 + exp_wx))  # 计算当前维度为i的梯度
            regular = self.learning_step*self.alpha*self.w[j]
            self.w[j] -= regular + self.learning_step * gradient  # 更新第i维的w
        return

    # 利用已经训练好的模型预测测试数据集的标签
    def predict_dataSet(self, features):
        labels = []  # 新建一个列表,用与储存预测的标签
        for feature in features:  # 遍历每一个测试数据集
            x = list(feature)  # 获取一个样本实例数据
            x.append(1)  # 对x进行拓展
            labels.append(self.perdict(x))  # 预测样本的取值
        return labels

    def save_model(self, path='./temp.pk'):
        with open(path, 'wb') as f:
            pkl.dump(self.w, f)
            print('保存模型成功')

    def load_model(self, path='./temp.pk'):
        if os.path.exists(path):
            f = open(path, 'rb')
            self.w = pkl.load(f)
            print('加载模型成功')
        else:
            print('加载模型失败')

def measure(test_labels, test_predict):
    accuracy = accuracy_score(test_labels, test_predict)
    precision = precision_score(test_labels, test_predict)
    recall = recall_score(test_labels, test_predict)
    print('accuracy=',accuracy)
    print('precision=',precision)
    print('recall=',recall)
    print('F1=', 2*recall*precision/(recall+precision))

if __name__=="__main__":
    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('data/mini_train_binary.csv', header=0)  # 从数据集里面读取数据, header=0说明第0行作为列名
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
    lr.train(train_features, train_labels, test_features, test_labels)  # 训练
    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    lr.save_model()

    print('Start predicting')
    test_predict = lr.predict_dataSet(test_features)  # 开始预测
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')

    # 计算准确率，计算召回率，计算F1测度
    measure(test_labels, test_predict)





