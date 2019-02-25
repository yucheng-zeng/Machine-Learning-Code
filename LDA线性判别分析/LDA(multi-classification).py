import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA 降维
def LDA_dimensionality(X, y, k):
    '''
    :param X: 数据集
    :param y: 标记
    :param k: 目标维数
    :return:
    '''
    label_ = list(set(y))  # 不偶去标记集合
    X_classify = {}  # 分类数据集X的字典, 键储存标记, 值储存不同类别的X样本点
    # 按标记不同, 将X划分为不同的集合
    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    mju = np.mean(X, axis=0)  # 计算所有样例的均值向量
    mju_classify = {}  # 记录不同类别的X样本点集合的平均值

    for label in label_:
        mju1 = np.mean(X_classify[label], axis=0)  # 计算当前类别的X样本点集合的平均值
        mju_classify[label] = mju1  # 添加到字典之中

    # St = np.dot((X - mju).T, X - mju)

    Sw = np.zeros((len(mju), len(mju)))  # 记录类内散度矩阵
    for i in label_:
        Sw += np.dot((X_classify[i] - mju_classify[i]).T,
                     X_classify[i] - mju_classify[i])  # 计算类内散度矩阵

    # Sb=St-Sw
    Sb = np.zeros((len(mju), len(mju)))  # 记录类间散度矩阵
    for i in label_:
        # 计算类间散度矩阵
        Sb += len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
            (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))

    # eig_vals 是特征值, eig_vec是特征矩阵
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵
    childEnergyRating = calChildSigmaEnergy(eig_vals, k)
    print(eig_vals)
    print('降维后能量占比=%s'%childEnergyRating)
    sorted_indices = np.argsort(eig_vals)  # 从小到大对特征值进行排序
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征值最大的特征向量
    return topk_eig_vecs  # 返回前k个特征值最大的特征向量

# 选取sigma前n个奇异值的能量值
def calChildSigmaEnergy(Sigma, k):
    Sig = Sigma**2  # 处理Sigma
    totalEnergy = sum(Sig)  # 计算总能量
    childEnergy = sum(Sig[:k])  # 计算钱i个奇异值的总能量
    # print(Sig)
    # print(Sigma)
    # print(Sigma[:k])
    # print(totalEnergy)
    # print(childEnergy)
    return float(childEnergy)/totalEnergy

if '__main__' == __name__:
    iris = load_iris()  # 加载sklearn.datasets自带数据集合
    X = iris.data  # 获取该数据集的特征集
    y = iris.target  # 获取该数据的标签

    #print('X=', X)
    #print('Y=', y)

    W = LDA_dimensionality(X, y, 2)  # 计算获得特征矩阵
    #print('W=', W)
    X_new = np.dot((X), W)  # 获得降维以后的X值
    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

    '''
    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    print(X_new)
    plt.figure(2)
    plt.title('LDA function in sklearn')
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
    '''

    plt.show()