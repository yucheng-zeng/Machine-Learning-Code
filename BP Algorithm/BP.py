from numpy import *

# 加载数据
def loadDataSet(fileName):
    dataMat = []  # 记录样本的特征值
    labelMat = []  # 记录样本的标记
    with open(fileName, 'r') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            if int(lineArr[2]) == 0:
                labelMat.append([1.0, 0.0])
            else:
                labelMat.append([0.0, 1.0])
    return mat(dataMat), mat(labelMat)

# 定义阶跃函数
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

'''
标准bp算法
每次更新都只针对单个样例，参数更新得很频繁s
dataSet 训练数据集
labels 训练数据集对应的标签
标签采用one-hot编码(一位有效编码)，例如类别0对应标签为[1,0],类别1对应标签为[0,1]
alpha 学习率
num 隐层数，默认为1层, 设置数值过大容易过拟合
eachCount 每一层隐层的神经元数目, 设置数值过大容易过拟合
repeat 最大迭代次数, 设置数值过大容易过拟合
算法终止条件：达到最大迭代次数或者相邻10次迭代的累计误差的差值不超过0.001
'''
def bp(dataSet, labels, alpha = 0.01, num = 1, eachCount = 3, repeat = 500):
    dataSet = mat(dataSet)
    m, n = shape(dataSet)  # 获取数据集的个数以及维度
    if len(labels) == 0:  # 若果没有训练数据退出
        print('no train data! ')
        return
    yCount = shape(labels[0])[1]  # 输出神经元的数目, 数目等于标签的种类个数
    print('yCount=%s'%yCount)
    # 创建一个n×eachCount的矩阵, 输入层到第一层隐层的w值和阈值，每列第一个为阈值, n为输入层属性的个数, 一共有eachCount个隐层神经元
    firstWMat = mat(random.sample((n + 1, eachCount)))
    print('firstWMat=%s'%firstWMat)
    hideWArr = random.sample((num - 1, eachCount + 1, eachCount))  # 隐藏层的w值和阈值，每列第一个为阈值
    print('hideWArr=%s'%hideWArr)
    lastWMat = mat(random.sample((eachCount + 1, yCount)))  # 最后一个隐层到输出神经元的w值和阈值，每列第一个为阈值
    # 隐层的输入, num代表右多少个隐层, hideInputs[i,j] 表示第i+1层(i和j都是从0开始)层隐层的第j+1个神经元的输入
    hideInputs = mat(zeros((num, eachCount)))
    # 隐层的输出, num代表右多少个隐层, hideOutputs[i,j] 表示第i+1层(i和j都是从0开始)层隐层的第j+1个神经元的输出
    hideOutputs = mat(zeros((num, eachCount + 1)))
    hideOutputs[:, 0] = -1.0    # 初始化隐层输出的每列第一个值为-1，即下一层功能神经元的阈值对应的输入恒为-1
    hideEh = mat(zeros((num, eachCount)))     # 隐层的梯度项
    yInputs = mat(zeros((1, yCount)))   # 输出层的输入
    i = 0   # 迭代次数
    old_ey = 0  # 前一次迭代的累积误差
    sn = 0  # 相邻迭代的累计误差的差值不超过0.001的次数
    while i < repeat:
        for r in range(len(dataSet)):  # 遍历每一个样本
            line = dataSet[r]  # 获取当前的样本
            # 根据输入样本计算隐层的输入和输出
            xMat = mat(insert(line, 0, values=-1.0, axis=1))  # 生成一个向量, 每个向量的第一维为-1, 之后的维度对应着样本点的特征值
            print(xMat)
            hideInputs[0, :] = xMat * firstWMat  # 计算隐层的输入
            hideOutputs[0, 1:] = sigmoid(hideInputs[0, :])  # 计算隐层的输出
            print('len(hideInputs)=%s'%len(hideInputs))
            # 隐藏层之间的值传递
            for j in range(1, len(hideInputs)):  #
                print('sb')
                hideInputs[j, :] = hideOutputs[j - 1, :] * mat(hideWArr[j - 1, :, :])
                hideOutputs[j, 1:] = sigmoid(hideInputs[j, :])

            # 根据与输出层连接的隐层的输出值计算输出层神经元的输入
            yInputs[0, :] = hideOutputs[len(hideInputs) - 1, :] * lastWMat
            # 计算近似输出
            yHead = sigmoid(yInputs)
            # 获取真实类别
            yReal = labels[r]
            # 计算输出层神经元的梯度项
            gj = array(yHead) * array(1 - yHead) * array((yReal - yHead))
            # 计算隐层的梯度项
            lastSumWGj = lastWMat[1:, :] * mat(gj).T  # lastWMat[1:, :]这里是b集合,其中的元素是bh,lastSumWGj是w的集合, 其中的元素是whj
            bMb = multiply(hideOutputs[num - 1, 1:], 1 - hideOutputs[num - 1, 1:])  # 计算bh(1-bh)
            hideEh[num - 1, :] = multiply(bMb, lastSumWGj.T)  # 计算eh
            # 计算隐藏层的梯度, 自上往下计算
            for q in range(num - 1):
                index = num - 2 - q
                hideSumWEh = mat(hideWArr[index])[1:, :] * hideEh[index + 1].T
                bMb = multiply(hideOutputs[index, 1:], 1 - hideOutputs[index, 1:])
                hideEh[index, :] = multiply(bMb, hideSumWEh.T)

            # 更新各层神经元的连接权和阈值
            lastWMat[:,:] = lastWMat[:,:] + alpha * hideOutputs[num - 1].T * mat(gj)
            firstWMat[:,:] = firstWMat[:,:] + alpha * xMat[0, :].T * mat(hideEh[0, :])
            # 更新隐层的连接权和阈值
            for p in range(num - 1):
                hideWArrMat = mat(hideWArr[p])
                hideWArrMat[:, :] = hideWArrMat[:, :] + alpha * hideOutputs[p].T * mat(hideEh[p + 1, :])
                hideWArr[p] = array(hideWArrMat)
        print('repeat: %d' % i)
        # 计算迭代累积误差
        ey = (yHead - yReal) * (yHead - yReal).T
        # 判断是否达到迭代终止条件
        if abs(ey - old_ey) < 0.001:
            sn = sn + 1
            old_ey = ey
            if sn >= 10:
                break
        else:
            sn = 0
            old_ey = ey
        i = i + 1
    return firstWMat, hideWArr,lastWMat, old_ey  # 返回输入层, 隐藏层, 输出层, 训练误差


# 获取y的近似输出
def getYHead(inX, yCount, firstWMat, hideWArr, lastWMat):
    num = len(hideWArr) + 1  # 隐层数目
    eachCount = shape(hideWArr)[2]  # 每一层隐层的神经元数目
    hideInputs = mat(zeros((num, eachCount)))  # 隐层的输入
    hideOutputs = mat(zeros((num, eachCount + 1)))  # 隐层的输出
    hideOutputs[:, 0] = -1.0  # 初始化隐层输出的每列第一个值为-1, 即下一层功能神经元的阈值对应的输入恒为-1
    yInputs = mat(zeros((1, yCount)))  # 输出层的输入

    # 计算隐层的输入
    xMat = mat(insert(inX, 0, values=-1.0, axis=1))
    hideInputs[0, :] = xMat * firstWMat  # 隐藏层的输入
    hideOutputs[0, 1:] = sigmoid(hideInputs[0, :])  # 隐藏层的输出
    for j in range(1, len(hideInputs)):  # 遍历每一个隐藏层
        hideInputs[j, :] = hideOutputs[j - 1, :] * mat(hideWArr[j - 1, :, :])  # 计算每一个隐藏层的输入
        hideOutputs[j, 1:] = sigmoid(hideInputs[j, :])  # 计算每一个隐藏层的输出

    # 计算输出层的输入
    yInputs[0, :] = hideOutputs[len(hideInputs) - 1, :] * lastWMat

    # 计算近似输出
    yHead = sigmoid(yInputs)
    return yHead

def calErrorRating(firstWMat, hideWArr, lastWMat,dataTestSet, testLabels):
    labelsHead = []  # 记录测试样本的标签
    for line in dataTestSet:  # 遍历每一个样本点, 计算每一个样本点在模型下的输出
        yHead = getYHead(line, 2, firstWMat, hideWArr, lastWMat)
        labelsHead.append(yHead)
    errorCount = 0

    for i in range(len(testLabels)):
        if testLabels[i, 0] == 1:
            yReal = 0
        else:
            yReal = 1

        if labelsHead[i][0, 0] > labelsHead[i][0, 1]:
            yEs = 0
        else:
            yEs = 1
        if yReal != yEs:
            print('error when test: [%f, %f], real: %d, error: %d' % (
            dataTestSet[i][0, 0], dataTestSet[i][0, 1], yReal, yEs))
            errorCount = errorCount + 1
    print('error rate: %f' % (float(errorCount) / len(dataTestSet)))
    return float(errorCount) / len(dataTestSet) , labelsHead

# 测试性能
def test():
    dataSet, labels = loadDataSet('dataSet.txt')  # 加载数据
    firstWMat, hideWArr, lastWMat, ey = bp(dataSet, labels)  # 训练模型
    dataTestSet, testLabels = loadDataSet('testSet.txt')  # 加载测试集合
    errorRate, labelsHead = calErrorRating(firstWMat, hideWArr, lastWMat, dataTestSet, testLabels)
    return labelsHead


if __name__ == '__main__':
    test()
