import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载iris数据集
def load_data():
    iris = datasets.load_iris()  # 加载数据
    # 使用交叉验证的方法，把数据集分为训练集合测试集
    # 将数据集拆分为训练集和测试集
    X_train, X_test, y_train_label, y_test_label = train_test_split(iris.data, iris.target, test_size=0.10, random_state=0)
    return X_train, X_test, y_train_label, y_test_label


# 读取数据
def loadDataSet(filename):
    dataMat=[]  # 记录样本点特征
    labelMat=[]  # 记录样本的标记
    fr=open(filename)
    for line in fr.readlines():
        lineArr=line.strip().split(',')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    X_train, X_test, y_train_label, y_test_label = train_test_split(dataMat, labelMat, test_size=0.10, random_state=2333) # 分割数据
    return X_train, X_test, y_train_label, y_test_label


# 使用SVC考察SVM的预测能力
def test_LinearSVC(X_train, X_test, y_train_label, y_test_label, method='linear'):

    # 选择线性模型
    #cls = svm.LinearSVC()
    cls = svm.SVC(kernel=method)

    # 把数据交给模型训练
    cls.fit(X_train,y_train_label)

    #print('Coefficients:%s'%cls.coef_)
    #print('Intercept:%s'%cls.intercept_)
    score = cls.score(X_test, y_test_label)  # 测试精确度
    print('Score: %.2f' % score)
    return cls


# 画图
def paintView(X_train, X_test, y_train_label, y_test_label):
    Xp = []  # 记录正类
    Xn = []  # 记录负类
    for index in range(len(X_train)):
        if y_train_label[index] == 1:
            Xp.append(X_train[index])
        elif y_train_label[index] == -1:
            Xn.append(X_train[index])

    for index in range(len(X_test)):
        if y_test_label[index] == 1:
            Xp.append(X_train[index])
        else:
            Xn.append(X_train[index])
    Xp = np.array(Xp)
    Xn = np.array(Xn)
    plt.scatter(Xp[:, 0], Xp[:, 1], color='k')
    plt.scatter(Xn[:, 0], Xn[:, 1], color='r')
    plt.show()


if __name__=="__main__":
    '''
    X_train, X_test, y_train_label, y_test_label = load_data()  # 生成用于分类的数据集
    cls = test_LinearSVC(X_train, X_test, y_train_label, y_test_label, method='linear')  # 调用 SVC, 使用线性模型
    result = cls.predict(X_test)
    print('result=%s' % result, '\ny_test_label=%s' % y_test_label)
    score = np.corrcoef(result, y_test_label)[0][1]
    print('score = %s' % score)
    '''

    X_train, X_test, y_train_label, y_test_label=loadDataSet('trainDataSet.txt')  # 生成用于分类的数据集
    cls = test_LinearSVC(X_train, X_test, y_train_label, y_test_label, method='rbf')  # 调用 SVC, 使用核函数
    result = cls.predict(X_test)
    print('result=%s'%result,'\ny_test_label=%s'%y_test_label)
    score = np.corrcoef(result,y_test_label)[0][1]
    print('score = %s' % score)
    paintView(X_train, X_test, y_train_label, y_test_label)
