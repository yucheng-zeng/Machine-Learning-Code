# 分类与回归树 (classification and regression tree, CART)
# 这里是用最小二乘法求回归树：在回归问题中，特征选择及最佳划分特征值的依据是：划分后样本的均方差之和最小
# 步骤：1、特征选择 2、回归树的生成 3、剪枝
# 适用于连续型数据, 数值型数据

from numpy import *
from paintTree import createPlot


# 导入数据集
def loadData(filaName):
    dataSet = []
    fr = open(filaName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')  # 分割字符串,生成一个列表
        theLine = list(map(float, curLine))  # 将列表里面的元素全部投影为浮点型数据
        dataSet.append(theLine)  # 将列表里面的数据增加到数据集之中
    return dataSet

# 计算总的方差
def GetAllVar(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]  # 计算第二维的方差, 然后乘以实例个数

# 根据给定的特征、特征值划分数据集
def dataSplit(dataSet,feature,featNumber):
    dataL = dataSet[nonzero(dataSet[:,feature] > featNumber)[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如大于, 则返回这个样本数据
    dataR = dataSet[nonzero(dataSet[:,feature] <= featNumber)[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如小于等于, 则返回这个样本数据
    return dataL, dataR  # 分别返回划分后左边的子数据集, 以及右边的子数据集

# op = [m,n], 用于表示停止条件, 避免回归树的结构过于复杂, 造成过拟合现象, 这里称为预剪枝
# m表示剪枝前总方差与剪枝后总方差差值的最小值； n: 数据集划分为左右两个子数据集后，子数据集中的样本的最少数量；
# 函数目的：特征划分
def choseBestFeature(dataSet,op = [1,4]):          # 三个停止条件可否当作是三个预剪枝操作
    # 获取数据集中的最后一维的数据,做成列表
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  # 停止条件 1, 如果只有一个样本点, 停止, 返回的最佳划分特征值会为当前数据集标签的平均值
        regLeaf = mean(dataSet[:,-1])  # 的最佳划分值为当前所有标签的均值
        return None, regLeaf           # 返回标签的均值作为叶子节点
    Serror = GetAllVar(dataSet)  # 获取划分前数据集的总方差
    BestFeature = -1  # 最佳划分特征
    BestNumber = 0  # 最佳划分特征的下标
    lowError = inf  # 划分之后的数据集的方差
    m,n = shape(dataSet)  # 获取样本个数m, 特征个数n
    for i in range(n-1):    # 遍历每一个特征值
        # 获取数据集中的第i维的数据,做成集合,避免重复
        # 遍历这些数据,找到算法之中的(j,s)
        # 确定j,遍历找到最优切分点s
        # 遍历j找到最优切分变量j(最佳特征的维度)
        # 确定一对最有切分(j,s)
        for j in set(dataSet[:,i].T.tolist()[0]):  # 以第i个样本作为切分变量, 最优切分点s
            #print('i am here')
            #print(dataSet[:,i])
            dataL,dataR = dataSplit(dataSet,i,j)  # 切分数据集
            if shape(dataR)[0]<op[1] or shape(dataL)[0]<op[1]:
                continue  # 如果所给的划分后的数据集中样本数目甚少,放弃这个划分对,则直接跳出
            tempError = GetAllVar(dataL) + GetAllVar(dataR)  # 记录划分后两个数据集的方差
            if tempError < lowError:  # 如果划分后方差比划分前小
                #  记录当前最有划分对
                lowError = tempError
                BestFeature = i
                BestNumber = j
    if Serror - lowError < op[0]:  # 停止条件2,如果所给的数据划分前后的差别不大，则停止划分,则返回的最佳特征为空,返回当前最佳划分值为当前所有标签的均值
        return None,mean(dataSet[:,-1])

    dataL, dataR = dataSplit(dataSet, BestFeature, BestNumber)  # 按最优划分对划分数据集, 用于合法性判断
    if shape(dataR)[0] < op[1] or shape(dataL)[0] < op[1]:  # 停止条件3,如果所给的划分后的数据集中样本数目甚少
                                                            # 则返回的最佳特征为空,返回的当前最佳划分特征值会为当前数据集标签的平均值
        return None, mean(dataSet[:, -1])

    return BestFeature, BestNumber  # 返回最佳划分对

# 决策树生成
def createTree(dataSet,op=[1,4]):
    bestFeat, bestNumber = choseBestFeature(dataSet,op)
    if bestFeat==None:  # 递归结束条件, 没有合法划分对
        return bestNumber
    regTree = {}  # 存储回归树
    regTree['spInd'] = bestFeat  # 存储最优划分变量(最优划分特征)
    regTree['spVal'] = bestNumber  # 存储最优划分点
    dataL, dataR = dataSplit(dataSet, bestFeat, bestNumber) # 按最优划分对划分数据集
    regTree['left'] = createTree(dataL,op)  # 递归调用, 生成左子树
    regTree['right'] = createTree(dataR,op)  # 递归调用, 生成右子树
    return regTree




# 以下是进行剪枝数处理
# 思想：用与当前树一样的结构对测试集进行处理, 生成一结构一样的树(称为测试树或者测试集),
#      1、计算测试树与当前树叶结点的误差
#      2、计算测试树与当前树的叶结点的父结点的误差
#      3、如果 1 > 2, 对叶结点进行剪枝, 以叶结点的父结点作为最新的叶结点; 否则, 不做改变
#      4、重复1～3, 继续往根结点进行回溯,直到遇到根结点后, 停止剪枝

# 用于判断所给的节点是否是子结点
def isTree(Tree):
    return (type(Tree).__name__ == 'dict')  # 叶结点子后没有子树了, 叶结点不是字典

# 计算两个子节点的均值
def getMean(Tree):
    if isTree(Tree['left']):
        Tree['left'] = getMean(Tree['left'])
    if isTree(Tree['right']):
        Tree['right'] = getMean(Tree['right'])
    return (Tree['left']+Tree['right'])/2.0


# 后剪枝
def pruneTree(Tree,testData):

    if shape(testData)[0]==0: #若果没有测试数据,则对树进行塌陷处理(即返回树平均值)
        return getMean(Tree)

    # 直到树的叶结点, 停止
    if isTree(Tree['left']) or isTree(Tree['right']):  # 若测试集存在子树
        dataL,dataR = dataSplit(testData,Tree['spInd'],Tree['spVal'])  # 按自身的树结构对测试数据进行同等结构的划分
    if isTree(Tree['left']):  # 如果自身还有左子树, 可继续进行剪枝处理
        Tree['left'] = pruneTree(Tree['left'], dataL)  # 用数据集的左子集对树进行剪枝
    if isTree(Tree['right']):  # # 如果自身还有右子树, 可继续进行剪枝处理
        Tree['right'] = pruneTree(Tree['right'], dataR)  # 用数据集的右子集对树进行剪枝

    # 往上回溯
    if not isTree(Tree['left']) and not isTree(Tree['right']):  # 判断是否到达了树的叶结点
        dataL,dataR = dataSplit(testData,Tree['spInd'],Tree['spVal'])  # 按自身的树结构对测试数据进行最后一次同等结构的划分, 这时测试数据也到了相应的叶结点

        # 计算同侧的测试集与自身树对应侧的树的标签的平均值的差, 作为误差值
        errorNoMerge = sum(power(dataL[:,-1]-Tree['left'],2)) + sum(power(dataR[:,-1]-Tree['right'],2))
        leafMean = getMean(Tree)  # 计算自身树叶结点的父结点标签的平均值
        errorMerge = sum(power(testData[:,-1]-leafMean,2))  # 计算测试集叶结点的父结点与自身树叶结点的父结点的平均值的误差值
        if errorNoMerge > errorMerge:  # 如果子结点的误差比父结点误差要大, 则剪去当前叶结点,以叶结点的父结点作为当前的最新的叶结点
            print("the leaf merge")
            return leafMean  # 返回新的叶结点
        else:
            return Tree  # 否则, 无需剪去叶结点
    else:
        print('sb')
        return Tree

# 预测, 给予某个值,将其正确分类归类到某一个特征集之中,并返回这个特征集的标签
def forecastSample(Tree,testData):
    if not isTree(Tree):  # 判断当前树是否到达了结点位置
        return float(Tree)  # 如果是返回结点的标签的均值
    # print"选择的特征是：" ,Tree['spInd']
    # print"测试数据的特征值是：" ,testData[Tree['spInd']]
    # 这个0表示第一个样本
    if testData[0,Tree['spInd']] > Tree['spVal']:  # 如果样本的点大于树的结点的均值, 往左向下搜索
        #print('sb')
        #print(testData[0,Tree['spInd']])
        if isTree(Tree['left']):  # 判断该结点是否存在子树, 若果存在继续向下搜索
            return forecastSample(Tree['left'],testData)
        else:  # 否则返回该结点的均值
            return float(Tree['left'])
    else:  # 如果样本的点小于等于树的结点的均值, 往右向下搜索
        if isTree(Tree['right']):  # 判断该结点是否存在子树, 若果存在继续向下搜索
            return forecastSample(Tree['right'],testData)   #
        else:  # 否则返回该结点的均值
            return float(Tree['right'])

# 输入当前树, 测试数据集, 返回的更新标签之后的测试数据集
def TreeForecast(Tree,testData):
    m = shape(testData)[0]  # 获取测试集样本点的个数
    y_hat = mat(zeros((m,1)))  # 创建也个m行,维度为1,元素全部为0的矩阵
    for i in range(m):  # 遍历样本,计算每个的预测值
        y_hat[i, -1] = forecastSample(Tree,testData[i])  # 输入当前树和样本点, 返回该样本点在该树之中的对应的最优划分变量,即是预测值
    return y_hat

#dataSet = loadData("ex2.txt")
#dataMat = mat(dataSet)
#regTree = createTree(dataMat,[1,6])
#print(regTree)
#print(getMean(regTree))
#print(dataMat[0,1])
#print(isTree(regTree))
#print(shape(dataMat)[0])
#print(nonzero(dataMat[:,1]>20))
#print(dataMat[:,-1])
#print(dataMat[:,-1].T.tolist()[0])


dataMat = loadData("ex2.txt")
dataMat = mat(dataMat)
op = [1,6]    # 参数1：剪枝前总方差与剪枝后总方差差值的最小值；参数2：将数据集划分为两个子数据集后，子数据集中的样本的最少数量；
theCreateTree = createTree(dataMat, op)
#print(theCreateTree)
# 测试数据
dataMat2 = loadData("ex2test.txt")
dataMat2 = mat(dataMat2)
y = dataMat2[:, -1]  # 保存原本数据集的标签值

# 剪枝相似度更好
#thePruneTree = pruneTree(theCreateTree, dataMat2)
#y_hat = TreeForecast(thePruneTree,dataMat2)
#print(corrcoef(y_hat,y,rowvar=0)[0,1])  # 计算原本数据集标签与预测后更新后的数据集的标签的误差
#print("剪枝后的后树：\n",thePruneTree)


y_hat = TreeForecast(theCreateTree,dataMat2)  # 获取预测数据集的标签
# r = corrcoef(x1,x2) 计算俩个矩阵相关度,输出结果为一个协方差（对称）矩阵
# 矩阵之中的值为0表示两个矩阵完全对应维度不相关; 矩阵之中的值1表示两个矩阵对应维度完全相关
# rowvar=0 计算两个矩阵的列相关度
print(corrcoef(y_hat,y,rowvar=0)[0,1])  # 计算原本数据集标签与预测后更新后的数据集的标签的误差

