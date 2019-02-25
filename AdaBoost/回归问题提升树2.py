import numpy as np
from sklearn.metrics import accuracy_score


class CBoostingTree(object):
    '''
    实现回归问题提升树(Boostring Tree)学习算法
    学习给定训练样本集的回归问题的提升树模型，考虑只用树桩作为基函数
    '''

    def __init__(self, train_samples, max_err):
        ''' class init function
        :param train_samples:训练样本集
        :param max_err:提升树拟合训练数据的平方损失误差允许的最大误差值
        '''
        self.Samples = train_samples  # 训练样本集
        self.max_e = max_err  # 提升树拟合训练数据的平方损失误差允许的最大误差值
        self.rsd = []  # 残差表 shape=(N, ), 提升树拟合训练数据，初始值为训练数据集的输出空间Y
        self.S = []  # shape=(1,N-1) 训练样本的切分点集合, segmentation point
        self.FX = []  # shape(M, 3)  e=[s,c1,c2]=一个回归树，m是回归树个数

        self.init_S()  # 切分数据
        self.build_boostingTree()  # 建立提升树
        return

    def init_S(self):
        '''计算所有可能切分点的集合，self.S'''
        X = self.Samples[:, 0]
        N = np.shape(self.Samples)[0]  # 获取训练数据样本的个数
        for i in np.arange(N)[0:-1]:  # 计算切分点
            self.S.append((X[i] + X[i + 1]) / 2.0)
        return self.S

    def build_boostingTree(self):
        '''提升树学习算法的迭代执行单元'''
        if 0 == np.shape(self.FX)[0]:  # 若果提升树的个数为0
            self.rsd = self.Samples[:, 1]  #　初始化残差表, 第一次的残差表为y的值
        X = self.Samples[:, 0]  # 初始化X
        Y = self.rsd  # 初始化Y
        C = []  # 记录R1和R2区域的平均回归值
        M = []  # 记录平方损失函数
        for i in range(np.shape(self.S)[0]):
            # RY[0] = [Y[0],...,Y[i]],RY[1] = [Y[i+1,...,Y[-1]]]
            RY1 = Y[0:i + 1]  # 获取右边区域的数据
            RY2 = Y[i + 1:]  # 获取左边区域的数据
            # 记录R1和R2区域的平均回归值
            C.append([np.mean(RY1), np.mean(RY2)])
            # 求切分点的回归误差平方和m
            m1_ufunc = lambda y: np.power(y - C[i][0], 2)
            m2_ufunc = lambda y: np.power(y - C[i][1], 2)
            m1_func = np.frompyfunc(m1_ufunc, 1, 1)
            m2_func = np.frompyfunc(m2_ufunc, 1, 1)
            cur_m = m1_func(RY1).sum() + m2_func(RY2).sum()
            M.append(cur_m)  # 记录平方损失函数

        # 生成回归树
        i = np.argmin(M)  # 获取平方损失函数切分的表
        s, c1, c2 = self.S[i], C[i][0], C[i][1]  # 获取最佳切分值, 以及切分之后左右两边的数据集合区域的平均值

        # 更新提升树
        self.FX.append([s, c1, c2])

        # 更新残差表(用提升树拟合训练数据的残差表)
        rsd_ufunc = lambda x, y: (x < s and [y - c1] or [y - c2])[0]
        rsd_func = np.frompyfunc(rsd_ufunc, 2, 1)
        #print('残差表%s'%rsd_func(X, Y))
        self.rsd = rsd_func(X, Y)

        # 计算损失函数的值
        # 计算用提升树拟合训练数据的平方损失误差
        e_ufunc = lambda r: np.power(r, 2)  # r为残差
        e_func = np.frompyfunc(e_ufunc, 1, 1)
        e = e_func(self.rsd).sum()

        # 判断是否终止生成提升树
        if e > self.max_e:  # 大于设置的误差阈值，则继续生成提升树
            print('需要继续提升树的学习，因为：提升树拟合训练数据的平方损失误差%f大约允许的最大误差值%f' % (e, self.max_e))
            return self.build_boostingTree()
        else:
            print('停止提升树的学习，提升树拟合训练数据的平方损失误差%f小于允许的最大误差值%f' % (e, self.max_e))
            print('BoostingTree succeed as:\n', np.array(self.FX))
            return self.FX

    def Regress(self, test_samples):
        '''提升树的回归函数
        使用学习得到的提升树self.FX，对给定的测试数据(集)test_samples进行回归
        :param test_samples:测试数据集
        :return 回归结果
        '''
        base_tree = lambda x, v, f1, f2: (x < v and [f1] or [f2])[0]
        reg_efunc = lambda x: sum([base_tree(x, v, f1, f2) for v, f1, f2 in self.FX])  # 遍历树, 计算平均值在相加
        reg_func = np.frompyfunc(reg_efunc, 1, 1)
        regress_result = reg_func(test_samples)
        return regress_result


def CBoostingTree_manual():
    # 训练数据集 Train Samples
    # train_samples[:,0]为输入空间，train_samples[:,1]为输出空间
    '''train_samples = np.array([[1, 5.56],
                              [2, 5.70],
                              [3, 5.91],
                              [4, 6.40],
                              [5, 6.80],
                              [6, 7.05],
                              [7, 8.90],
                              [8, 8.70],
                              [9, 9.00],
                              [10, 9.05]])'''

    train_samples = np.array([[1, 2],
                              [2, 5],
                              [3, 10],
                              [4, 17],
                              [5, 26],
                              [6, 37],
                              [7, 50],
                              [8, 65],
                              [9, 82],
                              [10, 101]])

    test_samples = np.array([0.51, 0.6, 1.55, 4.35, 9.99, 10.49])  # 测试集合的X坐标, 在训练数据的范围内预测结果较好
    #print(test_samples)
    bt = CBoostingTree(train_samples, max_err=0.2)  # 预测结果, max_err过小会出现过拟合现象

    ret = bt.Regress(test_samples)
    print('\ntest_samples:', test_samples)
    print('boostingTree regress result:\n', ret)

    test_samples = train_samples[:, 0]
    ret = bt.Regress(test_samples)
    print('\ntest_samples:', test_samples)
    print('boostingTree regress result:\n', ret)


    test_result = np.mat(np.zeros((np.shape(train_samples)[0],1)))
    i = 0
    for item in ret:
        test_result[i] = item
        i += 1

    # 计算两个矩阵的相关度, 既计算预测与实际数据之间差距
    score = np.corrcoef(test_result, train_samples[::,-1::], rowvar=0)[0, 1]
    print("\n\nThe accruacy socre is ", score)
    print('The error score is ', 1 - score)

if __name__ == '__main__':
    CBoostingTree_manual()