from numpy import *
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

'''
本程序只适用于连续性,二分类问题
'''
def loadSimpleData():
    # 创建数据集
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    # 创建类别标签
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # 返回数据集和标签
    return datMat, classLabels

def loadDataSet(fileName):
    """
    Function：   自适应数据加载函数

    Input：      fileName：文件名称

    Output： dataMat：数据集
                labelMat：类别标签
    """
    # 自动获取特征个数，这是和之前不一样的地方
    numFeat = len(open(fileName).readline().split('\t'))
    #初始化数据集和标签列表
    dataMat = []
    labelMat = []
    #打开文件
    fr = open(fileName)
    #遍历每一行
    for line in fr.readlines():
        #初始化列表，用来存储每一行的数据
        lineArr = []
        #切分文本
        curLine = line.strip().split('\t')
        #遍历每一个特征，某人最后一列为标签
        for i in range(numFeat-1):
            #将切分的文本全部加入行列表中
            lineArr.append(float(curLine[i]))
        #将每个行列表加入到数据集中
        dataMat.append(lineArr)
        #将每个标签加入标签列表中
        labelMat.append(float(curLine[-1]))
    #返回数据集和标签列表
    return dataMat, labelMat

def buildStump(dataArr, classLabels, D):
    """
    Function：   找到最低错误率的单层决策树

    Input：      dataArr：数据集
                classLabels：数据标签
                D：数据集权重向量

    Output： bestStump：分类结果
                minError：最小错误率
                bestClasEst：最佳单层决策树
    """

    dataMatrix = mat(dataArr)  # 初始化数据集
    labelMat = mat(classLabels).T  # 初始化数据标签
    #print(labelMat)
    m,n = shape(dataMatrix)  # 获取行列值
    numSteps = 10.0  # 初始化步数，用于在特征的所有可能值上进行遍历, 越大越正确, 但是需要的时间页越多
    bestStump = {}  # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst = mat(zeros((m,1)))  # 初始化类别估计值
    minError = inf # 将最小错误率设无穷大，之后用于寻找可能的最小错误率

    # 遍历数据集中每一个特征
    for i in range(n):
        # 获取数据集的最大最小值
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        # 根据步数求得步长
        #print(rangeMin)
        #print(rangeMax)
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历每个步长
        for j in range(0, int(numSteps) + 1):
            #遍历每个不等号
            #print('j=%s'%j)
            for inequal in ['lt', 'gt']:  # 记录想左分,还是想有分的效果好
                # 设定阈值
                threshVal = (rangeMin + float(j) * stepSize)  # 范围min ~ max
                #print(threshVal)
                # 通过阈值比较对数据进行分类
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                #初始化错误计数向量
                errArr = mat(ones((m,1)))
                # 如果预测结果和标签相同，则相应位置0
                errArr[predictedVals == labelMat] = 0
                # 计算权值误差，这就是AdaBoost和分类器交互的地方
                weightedError = D.T * errArr
                #打印输出所有的值
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))

                #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树，更新各项值
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i  # 最佳分类特征
                    bestStump['thresh'] = threshVal  #　最佳分类值
                    bestStump['ineq'] = inequal  # 最佳分类方向

    #　返回最佳单层决策树，最小错误率，类别估计值
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    """
    Function：   找到最低错误率的单层决策树

    Input：      dataArr：数据集
                classLabels：数据标签
                numIt：迭代次数

    Output： weakClassArr：单层决策树列表
                aggClassEst：类别估计值
    """
    weakClassArr = []  # 初始化列表，用来存放单层决策树的信息
    m = shape(dataArr)[0]  # 获取数据集行数
    D = mat(ones((m,1))/m)  # 初始化向量D每个值均为1/m，D包含每个数据点的权重
    # print(D)
    aggClassEst = mat(zeros((m,1)))  # 初始化列向量，记录每个数据点的类别估计累计值
    # print(aggClassEst)
    #开始迭代
    for i in range(numIt):
        # 利用buildStump()函数找到最佳的单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print(bestStump)
        #print("D: ", D.T)
        #根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        #保存alpha的值
        bestStump['alpha'] = alpha
        #print('a=%s'%alpha)

        #填入数据到列表
        weakClassArr.append(bestStump)
        #print("classEst: ", classEst.T)

        # 为下一次迭代计算数据权重D
        oldD = D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        #print('expon=%s'%expon)
        expon = exp(mat(expon))
        D = multiply(D, expon)
        D = D / D.sum()
        # 数据集权重变化太小,退出
        #if abs(sum(D-oldD))< 1e-4:
        #    break
        #print('D=%s'%D)
        # 累加类别估计值
        #print('1aggClassEst=%s' % aggClassEst)
        aggClassEst += alpha * classEst  # 几率多个分类对数据集合的分类值
        #print('2aggClassEst=%s' % aggClassEst)
        #print("aggClassEst: ", aggClassEst.T)
        #计算错误率，aggClassEst本身是浮点数，需要通过sign来得到二分类结果
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        #如果总错误率为0则跳出循环
        if errorRate == 0.0:
            break
    #返回单层决策树列表和累计错误率
    return weakClassArr  #　bestStump保存单节点树的所有信息
    #return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    """
    Function：   AdaBoost分类函数

    Input：      datToClass：待分类样例
                classifierArr：多个弱分类器组成的数组

    Output： sign(aggClassEst)：分类结果
    """
    #初始化数据集
    dataMatrix = mat(datToClass)
    #获得待分类样例个数
    m = shape(dataMatrix)[0]
    #构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m,1)))
    # 遍历每个弱分类器
    for i in range(len(classifierArr)):
        # 基于每个stumpClassify得到类别估计值
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        # 累加类别估计值
        aggClassEst += classifierArr[i]['alpha']*classEst
        #打印aggClassEst，以便我们了解其变化情况
        #print(aggClassEst)
    #返回分类结果，aggClassEst大于0则返回+1，否则返回-1
    return sign(aggClassEst)

# 基础分类器
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    Function：   通过阈值比较对数据进行分类

    Input：     dataMatrix：数据集
                dimen：数据集列数
                threshVal：阈值
                threshIneq：比较方式：lt，gt

    Output： retArray：分类结果
    """
    # 新建一个数组用于存放分类结果，初始化都为1, 维度等于实例样本个数
    retArray = ones((shape(dataMatrix)[0],1))
    #print('1retArray=%s'%retArray)
    # lt：小于 gt；大于；根据阈值进行分类，并将分类结果存储到retArray
    if threshIneq == 'lt':
        #print(dataMatrix[:, dimen] <= threshVal)
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
        #print('2retArray=%s' % retArray)
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    # 返回分类结果
    return retArray

# 将数据的字符串标签转化为支付类型
def exchangeStr(labels):
    if not is_number(labels[0]):
        features = set(labels)
        length = len(labels)
        numbers = 0.
        for feature in features:
            numbers += 1.
            for i in range(0, length):
                if labels[i] == feature:
                    labels[i] = numbers



# 判断一个字符串是否是数据, 用于区分是连续型数据还是离散型数据
def is_number(label):
    try:
        float(label)
        return True
    except ValueError:
        return False


if __name__ == '__main__':

    #datMat, classLabels = loadSimpleData()
    #classifierArr = adaBoostTrainDS(datMat, classLabels, 30)
    #print(adaClassify([2, 1], classifierArr))
    #print(classLabels)

    '''
    datMat, classLabels = loadDataSet('horseColicTraining2.txt')
    classifierArr = adaBoostTrainDS(datMat, classLabels, 20)
    datMatTest, classLabelsTest = loadDataSet('horseColicTest2.txt')
    retarray = adaClassify(datMatTest, classifierArr)
    score = accuracy_score(classLabelsTest,retarray)
    print("The accruacy socre is ", score)
    print('The error score is ', 1 - score)
    '''

    '''
    raw_data = pd.read_csv('train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
                                imgs, labels, test_size=0.33, random_state=23323)
    classifierArr = adaBoostTrainDS(train_features, train_labels, 10)
    retarray = adaClassify(test_features, classifierArr)
    score = accuracy_score(test_labels, retarray)
    print("The accruacy socre is ", score)
    print('The error score is ', 1 - score)
    '''


    raw_data = pd.read_csv('Iris.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # 选取 3/4 数据作为训练集， 1/4 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.20, random_state=23333)
    classifierArr = adaBoostTrainDS(train_features, train_labels, 10)

    retarray = adaClassify(test_features, classifierArr)
    score = accuracy_score(test_labels, retarray)
    print(test_labels)
    print(retarray)
    print("The accruacy socre is ", score)
    print('The error score is ', 1 - score)
