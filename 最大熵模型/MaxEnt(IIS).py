from collections import defaultdict
import numpy as np


class maxEntropy(object):
    def __init__(self):
        self.trainset = []  # 训练数据集
        self.features = defaultdict(int)  # 用于获得(标签，特征)键值对
        self.labels = set([])  # 标签
        self.w = []  # 参数向量

    def loadData(self, fName):
        for line in open(fName):
            fields = line.strip().split()  # 分割数据
            if len(fields) < 2:  # 过滤数据
                continue
            label = fields[0]  # 第一列为标签
            self.labels.add(label)  # 获取label

            for f in set(fields[1:]):  # 对于每一个特征
                self.features[(label, f)] += 1  # 每提取一个（标签，特征）对，就自加1，统计该特征-标签对出现了多少次

            self.trainset.append(fields)  # 添加到训练集合
            self.w = [0.0] * len(self.features)  # 初始化权重
            self.lastw = self.w  # 迭代前的w

    # 对于该问题，M是一个定值，所以delta有解析解
    def train(self, max_iter=1000):  # 设置最大步数
        self.initP()  # 主要计算M以及联合分布在f上的期望

        # 下面计算条件分布及其期望，正式开始训练
        for i in range(max_iter):  # 计算条件分布在特征函数上的期望
            self.ep = self.EP()  # Ep 是模型P(X|Y)与经验分布P(X)的期望值
            self.lastw = self.w[:]  # 保存迭代前的w
            for i, w in enumerate(self.w):
                # 计算出theta(i)
                theta = (1.0 / self.M) * np.log(self.Ep_[i] / self.ep[i])
                self.w[i] += theta  # 更新w[i]
            if self.convergence():  # 判断是否要退出循环
                break

    def initP(self):
        # 获得M
        self.M = max([len(feature[1:]) for feature in self.trainset])  # M 值为最大特征数目
        self.size = len(self.trainset)  # 获取样本点个数
        print(self.size)
        self.Ep_ = [0.0] * len(self.features)  # 初始化期望值,维度等于每个样本的（标签，特征）个数
        # Ep_是特征函数f(x,y)关于经验分布P_(x,y)的期望值
        # 获得联合概率期望
        for i, feat in enumerate(self.features):  # i表示(标签，特征）对下标, feat表示(标签，特征）对
            print('i=%s,feat=%s' % (i, feat))  # 获得联合概率期望
            self.Ep_[i] += self.features[feat] / (1.0 * self.size)  # 获得联合概率期望
            # 更改键值对为（label-feature）-->id
            self.features[feat] = i
        # 准备好权重
        self.w = [0.0] * len(self.features) # 初始化w, 维度等于(标签，特征）
        self.lastw = self.w  # 迭代前的w

    # 　计算模型P(X|Y)与经验分布P(X)的期望值
    def EP(self):
        # 计算p（y|x）
        ep = [0.0] * len(self.features)  # 初始化模型P(X|Y)与经验分布P(X)的期望值, 维度等于(标签，特征)对的个数
        for record in self.trainset:  # 遍历数据集
            onefeatures = record[1:]  # 回去每一个样本的特征
            # 计算p（y|x）
            prob = self.calPyx(onefeatures)
            for f in onefeatures:  # 特征一个个来
                for pyx, label in prob:  # 获得条件概率与标签
                    if (label, f) in self.features:
                        id = self.features[(label, f)]  # 获取id
                        ep[id] += (1.0 / self.size) * pyx  # 计算相应的期望
        return ep

    # 获得最终单一样本每个特征的pyx
    def calPyx(self, onefeatures):
        # 传的onefeatures是单个样本的
        wlpair = [(self.calSumP(onefeatures, label), label) for label in self.labels] # 遍历标签,构建(标签，特征)对,计算期望值
        Z = sum([w for w, l in wlpair])  # 计算w的和
        prob = [(w / Z, l) for w, l in wlpair]
        return prob

    def calSumP(self, onefeatures, label):
        '''书本85页的分母Zw(x)'''
        sumP = 0.0
        #print('2%s'%onefeatures)
        # 对于这单个样本的feature来说，不存在于feature集合中的f=0所以要把存在的找出来计算
        for showedF in onefeatures:  # 获得单个样本的所有(标签，特征)键值对
            if (label, showedF) in self.features:
                index = self.features[(label, showedF)]  # 获取每一个标签对在w中的下标
                sumP += self.w[index]  # 获取每一个(标签，特征)键值对的w
        #print('sumP=%s'%np.exp(sumP))
        return np.exp(sumP)

    # 判断是否结束
    def convergence(self):
        for i in range(len(self.w)):
            if abs(self.w[i] - self.lastw[i]) >= 0.001:  # 如果w与lastw的一个维度都满足条件
                return False
        return True

    # 预测
    def predict(self, input):
        features = input.strip().split()
        prob = self.calPyx(features)
        prob.sort(reverse=True)
        return prob


if __name__ == '__main__':
    mxEnt = maxEntropy()
    mxEnt.loadData('gameLocation.dat')
    mxEnt.train()
    print(mxEnt.predict('Sunny'))