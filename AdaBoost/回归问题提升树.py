"""
提升树：基于二叉回归树的提升算法
程序暂考虑输入为一维的情况
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


class BoostingTree:
    def __init__(self, epsilon=1e-2):
        self.epsilon = epsilon  # 设置阀值
        self.cand_splits = []  # 候选切分点
        self.split_index = defaultdict(tuple)  # 由于要多次切分数据集，故预先存储，切分后数据点的索引
        self.split_list = []  # 最终各个基本回归树的切分点
        self.c1_list = []  # 切分点左区域取值
        self.c2_list = []  # 切分点右区域取值
        self.N = None  # 储存样本点个数
        self.n_split = None  # 记录所有切分点的个数

    def init_param(self, X_data):
        # 初始化参数
        self.N = X_data.shape[0]  # 过去样本点个数
        for i in range(1, self.N):
            self.cand_splits.append((X_data[i][0] + X_data[i - 1][0]) / 2)  # 计算所有可能的切分点
        self.n_split = len(self.cand_splits)
        for split in self.cand_splits:  # 遍历所有切分点
            left_index = np.where(X_data[:, 0] <= split)[0]  # 记录每个切分点的左边index
            right_index = list(set(range(self.N))-set(left_index))  # 记录切分点的右边index
            self.split_index[split] = (left_index, right_index)  # 记录切分点的左右的切分情况
        return

    def _cal_err(self, split, y_res):
        # 计算每个切分点的误差
        inds = self.split_index[split]  # 获取切分点下标
        left = y_res[inds[0]]  # 获取切分点左边的数据集合
        right = y_res[inds[1]]  # 获取切分点右边的数据集合

        c1 = np.sum(left) / len(left)  # 左侧平均值
        c2 = np.sum(right) / len(right)  # 右侧平均值
        y_res_left = left - c1  # 左侧的残差列表
        #print('y_res_left=%s'%y_res_left)
        y_res_right = right - c2  # 右侧的残差列表
        res = np.hstack([y_res_left, y_res_right])  # 获得一个完整的残差列表
        res_square = np.apply_along_axis(lambda x: x ** 2, 0, res).sum()  # 计算残差的平方和,既损失函数
        return res_square, c1, c2

    def best_split(self, y_res):
        # 获取最佳切分点，并返回对应的残差
        best_split = self.cand_splits[0]  # 初始化最佳切分点
        #print('best_split=%s'%best_split)
        min_res_square, best_c1, best_c2 = self._cal_err(best_split, y_res)  # 初始化损失函数值, 左侧与右侧数据集的平均值

        for i in range(1, self.n_split):  # 遍历所有切分点, 获取最佳切分点, 并且计算相关的值
            res_square, c1, c2 = self._cal_err(self.cand_splits[i], y_res)
            if res_square < min_res_square:
                best_split = self.cand_splits[i]
                min_res_square = res_square
                best_c1 = c1
                best_c2 = c2

        self.split_list.append(best_split)  # 记录最佳切分点
        self.c1_list.append(best_c1)  # 记录左侧数据集的平均值
        self.c2_list.append(best_c2)  # 记录右侧数据集的平均值
        return

    def _fx(self, X):
        # 基于当前组合树，预测X的输出值
        s = 0
        # 遍历每一棵子树, 累加, 计算出组合树对x的预测值
        for split, c1, c2 in zip(self.split_list, self.c1_list, self.c2_list):
            if X < split:
                s += c1
            else:
                s += c2
        return s

    def update_y(self, X_data, y_data):
        # 每添加一颗回归树，就要更新y,即基于当前组合回归树的预测残差
        y_res = []  # 初始化化残差表
        for X, y in zip(X_data, y_data):  # 将X_data, y_data打包成一个（x,y）元素对
            y_res.append(y - self._fx(X[0]))  # 计算每一个元素对应的残差值
        y_res = np.array(y_res)
        res_square = np.apply_along_axis(lambda x: x ** 2, 0, y_res).sum()  # 计算出残差的平方, 既是损失函数
        return y_res, res_square

    def fit(self, X_data, y_data):
        self.init_param(X_data)  # 初始化参数, 村里参数
        y_res = y_data
        while True:
            self.best_split(y_res)
            y_res, res_square = self.update_y(X_data, y_data)
            if res_square < self.epsilon:  # 当损失函数的值小于阀值, 退出循环
                break
        return

    def predict(self, X):
        return self._fx(X)


if __name__ == '__main__':
    # data = np.array(
    #     [[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40], [5, 6.80], [6, 7.05], [7, 8.90], [8, 8.70], [9, 9.00], [10, 9.05]])
    # X_data = data[:, :-1]
    # y_data = data[:, -1]
    # BT = BoostingTree(epsilon=0.18)
    # BT.fit(X_data, y_data)
    # print(BT.split_list, BT.c1_list, BT.c2_list)
    X_data_raw = np.linspace(-5, 5, 100)  # 生成x
    #print('X_data_raw=%s'%X_data_raw)
    X_data = np.transpose([X_data_raw])  # 转秩, 生成列矩阵
    #print('X_data=%s'%X_data)
    y_data = np.sin(X_data_raw)  # 对应与每一个x,生y
    BT = BoostingTree(epsilon=0.1)
    BT.fit(X_data, y_data)

    y_pred = [BT.predict(X) for X in X_data]  # 计算出预测值

    # 计算两个矩阵的相关度, 既计算预测与实际数据之间差距
    score = np.corrcoef(y_pred, y_data, rowvar=0)[0, 1]
    print("\n\nThe accruacy socre is ", score)
    print('The error score is ', 1 - score)

    p1 = plt.scatter(X_data_raw, y_data, color='r')
    p2 = plt.scatter(X_data_raw, y_pred, color='b')
    plt.legend([p1, p2], ['real', 'pred'])
    plt.show()