import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm



def createDataSet():
    np.random.seed(0)  # 设置随机数种子
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]  # 随机创建40个点的集合, +-[2, 2]表示第几类
    #print('X=%s'%X)
    Y = [0] * 20 + [1] * 20  # 前20个点是负类, 后20个点是正类
    #print('Y=%s'%Y)
    return X, Y

def SVM(X,Y):
    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)
    return clf

if __name__ == '__main__':
    X, Y = createDataSet()

    clf = SVM(X, Y)

    # 计算得出分离超平面
    w = clf.coef_[0]  # 获取w
    #print('w=%s'%w)
    a = -w[0] / w[1]  # 获取斜率
    xx = np.linspace(-5, 5)  # 获取50个从-5到5的取值
    #print('xx=%s'%xx)
    yy = a * xx - (clf.intercept_[0]) / w[1]  # 计算的簇分离超平面

    # plot the parallels to the separating hyperplane that pass through the
    # 支持向量
    b = clf.support_vectors_[0]  # 获得位于下侧的支持向量
    #print('clf.support_vectors_[0]=%s'%b)
    yy_down = a * xx + (b[1] - a * b[0])  # 计算位于下侧的分离超平面
    b = clf.support_vectors_[2]  # 获得位于上侧的支持向量
    #print('clf.support_vectors_[-1]=%s'%b)
    yy_up = a * xx + (b[1] - a * b[0])  # # 计算位于上侧的分离超平面

    # 绘图
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none')
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.axis('tight')
    plt.show()
