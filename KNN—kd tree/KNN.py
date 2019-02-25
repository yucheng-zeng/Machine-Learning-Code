from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  # 样本点矩阵
    labels = ['A','A','B','B']  # 每个样本对应的标签
    return group, labels

def classify(target, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 获取样本点的个树
    diffMat = tile(target, (dataSetSize, 1)) - dataSet  # 建立一个行数为样本个数,列数为1的，里面每一个元素都为目标点的矩阵
                                                        # 减去数据集, 得到每个维度的汉米顿距离
    sqDiffMat = diffMat**2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 矩阵的每行的每个维度的列相加,得到一个一位矩阵
    distances = sqDistances**0.5  # 开方, 得到目标点与每个样本点的欧氏距离
    sortedDistances = distances.argsort()  # 安从小到大的给矩阵排序,返回排序后的索引值
    classCount = {}  # 创建空字典, 用于保存标签以及标签出现的次数
    for i in range(k):  # 从前k个最近点找到标签次数出现最多的那个类, 作为目标点的类
        numOfLabel = labels[sortedDistances[i]]  # 获取索引值为i的那个类
        #print(numOfLabel)
        classCount[numOfLabel] = classCount.get(numOfLabel, 0) + 1  # 该类出现次数+1
        #print(classCount[numOfLabel])
    # 将classCount排序,按标签出现次数从大到小排序，返回一个列表
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print(sortedClassCount[0])
    return sortedClassCount[0][0]  # 返回出现次数最多的标签

target = [0,0]
groups, labels = createDataSet()
print(classify(target, groups, labels, 3))