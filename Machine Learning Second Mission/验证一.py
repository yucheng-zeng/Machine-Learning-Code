import numpy as np
import matplotlib.pyplot as plt
import binary_logistic as bl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_data(x0=-5, x1=5, n=40, loc=0, scale=2, label=0):
    x = np.arange(x0, x1, (x1 - x0) / n)
    y = []
    for xi in x:
        y.append(normal(xi, loc, scale))
    return np.array(x), np.array(y), np.array([label]*n)

def normal(x, loc, scale):
    return (1/(np.emath.sqrt(2*np.pi)*scale))*np.exp(-((x-loc)**2)/(2*scale*scale))

# 画图
def show(x, y):
    plt.plot(x, y, color='b', linestyle='', marker='.')

def to_matrix(x, y):
    mat = []
    for i in range(0, len(x)):
        mat.append([x[i], y[i]])
    return np.array(mat)

if __name__ == '__main__':
    x0, y0, label0 = create_data(x0=1, x1=5, n=1000, loc=3,scale=2, label=0)
    show(x0, y0)

    x1, y1, label1 = create_data(x0=6, x1=10, n=1000, loc=8, scale=2, label=1)
    show(x1, y1)

    x = np.hstack((x0, x1))
    y = np.hstack((y0, y1))
    labels = np.hstack((label0, label1))
    mat = to_matrix(x, y)

    print(mat)

    train_features, test_features, train_labels, test_labels = train_test_split(mat, labels, test_size=0.33, random_state=23323)

    lr = bl.LogisticRegression(max_iteration=1)

    print('Start training')
    lr.train(train_features, train_labels, test_features, test_labels)  # 训练

    print('Start predicting')
    test_predict = lr.predict_dataSet(test_features)  # 开始预测

    # 预测结果
    bl.measure(test_labels, test_predict)

    count = 0
    for i in test_predict:
        if i == 0:
            count+=1
    print(count)