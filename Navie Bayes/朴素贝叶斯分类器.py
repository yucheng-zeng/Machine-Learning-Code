import operator

def loadDataSet():
    X = {}
    x1 =     [ 1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3]
    x2 =     ['S','M','M','S','S','S','M','M','L','L','L','M','M','L','L']
    x3 =     ['A','B','B','A','A','B','B','B','C','C','C','C','C','C','A']
    labels = [-1, -1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1]
    X[1] = x1
    X[2] = x2
    X[3] = x3
    return X, labels

# 计算先验概率
def prior_probability(labels):
    py = {}  # 用于记录先验概率
    labelsSet = set(labels)
    for i in labelsSet:
        py[i] = (labels.count(i)+1)/(len(labels)+len(set(labels)))  # 拉普拉斯修正
    return py

# 计算条件概率
def conditional_probability(xj, value, labels):
    xcount = {}  # 记录出现次数
    labelsSet = set(labels)  # 记录标签集合
    for i in range(len(labels)):
        for label in labelsSet:
            if labels[i] == label and xj[i] == value:
                xcount[label] = xcount.get(label, 0) + 1  # 出现次数加一

    pxy = {}  # 记录条件概率
    for label in labelsSet:
        pxy[label] = (float(xcount[label])+1)/(labels.count(label)+len(set(xj)))  # 计算条件概率, 拉普拉斯修正
    return pxy

# 朴素贝叶斯分类器
def classify(X, inX, labels):
    pxy = {}  # 用于计算条件概率
    for i in range(1,len(X)+1):  # 计算每一个特征的条件概率
        pxy[i] = conditional_probability(X[i],inX[i-1],labels)  # 计算离散型的条件概率
    py = prior_probability(labels)  # 计算先验概率
    pVec = {}  # 用于记录贝叶斯分类器的结果
    labelsSet = set(labels)  # 记录标签集合
    # 计算贝叶斯分类器的结果
    for label in labelsSet:  # 遍历每一个标签的取值
        pVec[label] = py.get(label)
        for i in range(1,len(X)+1):
            pVec[label] = pVec[label] * pxy[i].get(label)
    print(pVec)
    sortedClassCount = sorted(pVec.items(), key=operator.itemgetter(1), reverse=True)  # 按结果从大到小排序
    return sortedClassCount[0][0]  # 取概率最大的标签


if __name__=='__main__':
    X, labels = loadDataSet()
    inX = [2, 'M', 'B']
    print(classify(X, inX, labels))