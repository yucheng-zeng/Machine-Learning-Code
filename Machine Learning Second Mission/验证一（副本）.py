import numpy as np
import matplotlib.pyplot as plt
import binary_logistic as bl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_data(mean, cov, m, y):
    x = np.random.multivariate_normal(mean, cov, m)
    y = np.array([y]*m).reshape((m,1))
    return x, y

def to_high_dimension(x, n):
    temp = x
    for i in range(2,n+1):
        x = np.c_[x,temp**i]
    return x

def normal(x, loc, scale):
    return (1/(np.emath.sqrt(2*np.pi)*scale))*np.exp(-((x-loc)**2)/(2*scale*scale))

# 画图
def show(x, y, color='b', marker='.'):
    plt.plot(x, y, color=color, linestyle='', marker=marker)

def to_matrix(x, y):
    mat = []
    for i in range(0, len(x)):
        mat.append([x[i], y[i]])
    return np.array(mat)


if __name__ == '__main__':
    mean1 = [2, 2]
    cov1 = 0.1*np.eye(2)

    # 不满足贝叶斯条件独立性假设
    cov1[0][1] = -cov1[0][0]*0.9
    cov1[1][0] = cov1[0][1]

    mean2 = [0, 0]
    cov2 = 0.1 * np.eye(2)

    # 不满足贝叶斯条件独立性假设
    cov2[0][1] = -cov2[0][0]*0.9
    cov2[1][0] = cov2[0][1]

    x1, y1 = create_data(mean1, cov1, 200, 1)
    x2, y2 = create_data(mean2, cov2, 200, 0)



    x = np.vstack((x1, x2))
    y = np.vstack((y1, y2))

    x = to_high_dimension(x, 2)

    train_features, test_features, train_labels, test_labels = train_test_split(x, y, test_size=0.33, random_state=23323)

    lr = bl.LogisticRegression(max_iteration=10)

    print('Start training')
    lr.train(train_features, train_labels, test_features, test_labels)  # 训练

    print('Start predicting')
    test_predict = lr.predict_dataSet(test_features)  # 开始预测

    # 预测结果
    bl.measure(test_labels, test_predict)
