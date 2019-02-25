from numpy import *
from paintTree import createPlot
import operator
import re
import copy
# 从文本之中读取数据
def readDataFromFile(filename):
    labels = []
    dataSet = []
    with open(filename) as fileObject:
        title = fileObject.readline().strip()
    labels = title.split('\t')  #
    labels = [item.strip() for item in labels]  # 获取标签
    with open(filename) as fileObject:
        lines = fileObject.readlines()
    for line in lines:
        sublist = line.split('\t')
        sublist = [item.strip() for item in sublist]
        dataSet.append(sublist)  # 将每个实体加入到数据集之中
    del dataSet[0]  # 删除第一行, 即删除标签
    return dataSet, labels

# 计算数据集的基尼指数
def calGini(dataSet):
    numEntries = len(dataSet)  # 获取数据集的样本个数
    labelCounts = {}  # 用于保存标签,及其出现次数
    for featVec in dataSet:  # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1]  # 获取每个实例的最后一维, 即同意与否
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 初始化
        labelCounts[currentLabel] += 1  # 记录标签出现次数
    Gini = 1.0  # 初始化基尼指数
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 数据集中标签为特定值的样本占的样本总数的概率
        Gini -= prob*prob  # 计算基尼指数
    return Gini

# 对离散变量划分数据集，取出该特征取值为value的所有样本, 添加到数据集当中
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:  # 从数据集中获取每一个实例
        if featVec[axis]==value:  # 如果实例的特征与参数特征相同, 提出这个样本点
            #print('I AM HERE')
            # 要将这个特征从提取出来的样本点中剔除
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            #print(reducedFeatVec)
            #print(featVec)
            retDataSet.append(reducedFeatVec)  # 将这个处理过实例纳入新的数据集
    return retDataSet

'''
# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet
'''
'''
# 对连续变量划分数据集,同时划分出小于featNumber的数据样本和大于featNumber的数据样本集
def splitContinuousDataSet(dataSet,feature,featNumber):
    if not is_number(featNumber):
        dataL = dataSet[nonzero(dataSet[:,feature] <= featNumber)[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如大于, 则返回这个样本数据
        dataR = dataSet[nonzero(dataSet[:,feature] > featNumber)[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如小于等于, 则返回这个样本数据
        return dataL, dataR  # 分别返回划分后左边的子数据集, 以及右边的子数据集
    else:
        dataL = dataSet[nonzero(float(dataSet[:, feature]) <= float(featNumber))[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如大于, 则返回这个样本数据
        dataR = dataSet[nonzero(float(dataSet[:, feature]) > float(featNumber))[0],:]  # 将数据集的样本中维度为feature的数据与featNumber比较, 如小于等于, 则返回这个样本数据
        return dataL, dataR  # 分别返回划分后左边的子数据集, 以及右边的子数据集
'''
# 对连续变量划分数据集，direction规定划分的方向, 0表示取大于value的样本, 1表示取大于等于value的样本
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet,axis,value,direction):
    retDataSet=[]
    for featVec in dataSet:
        if is_number(value):  # 如果这个划分值是数字, 则说明这里要划分连续性数据, 要将数值转换成浮点型再进行比较
            featVec[axis] = float(featVec[axis])
        if direction==0:
            if featVec[axis]>value:
                # 要将这个特征从提取出来的样本点中剔除
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)  # 将这个处理过实例纳入新的数据集
        else:
            if featVec[axis]<=value:
                # 要将这个特征从提取出来的样本点中剔除
                reducedFeatVec=featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)  # 将这个处理过实例纳入新的数据集
    return retDataSet


# 判断一个字符串是否是数据, 用于区分是连续型数据还是离散型数据
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet, labels, dataSet_full, labels_full):
    numFeatures = len(labels) - 1  # 获得当前剩余用于分类的特征数目, 最后一维不能当作特征值
    bestGiniIndex = Inf  # 记录基尼指数最小值对应的特征的下标
    bestFeature = -1  # 记录基尼指数最小值对应的特征的值
    bestSplitDict = {}  # 记录基尼指数最小值对应的特征的所有可能的取值
    for i in range(numFeatures):  # 遍历所有特征值, 找到最优划特征, 与对连续性数据还需要找到该特征值之下对应的最优划分点
        featList = [example[i] for example in dataSet]  # 在数据集中获取每个样本特征i的取值

        # 如果这个样本的数据是连续型数据(数值可以是认为连续型数据),要按连续型数据方式对特征进行处理,
        # 这时计算基尼指数方式要发生变化, 要设定某一个值划分, 计算划分前后两个子数据集对总数据集的概率
        if is_number(featList[0]):
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)  # 对这个特征里的数据进行从小到大排序
            splitList = []  # 记录最优划分点
            for j in range(len(sortfeatList) - 1):  # 记录每两个邻近特征取值的中位数, 用于找优切分点
                splitList.append( ( float(sortfeatList[j]) + float(sortfeatList[j + 1]) ) / 2.0)

            bestSplitGini = Inf  # 记录基尼指数
            slen = len(splitList)  # 数据集样本个数
            # 求用第j个候选划分点划分时，得到的基尼指数，并记录最佳划分点
            for j in range(slen):
                value = splitList[j]  # 获取划分点的取值
                newGiniIndex = 0.0  # 初始化基尼指数
                #print(value)
                #print(i)
                dataL = splitContinuousDataSet(dataSet, i, value, 0)  # 划分数据集,获得左子数据集 以及 获得右子数据集
                dataR = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(dataL) / float(len(dataSet))  # 计算左子集占总数据集的概率
                newGiniIndex += prob0 * calGini(dataL)    # 计算基尼指数
                prob1 = len(dataR) / float(len(dataSet))  # 计算右子集占总数据集的概率
                newGiniIndex += prob1 * calGini(dataR)    # 计算基尼指数
                if newGiniIndex < bestSplitGini:
                    bestSplitGini = newGiniIndex  # 更新的最小基尼指数
                    bestSplit = j  # 并记录当前最佳划分点
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]  # 记录连续型数据的最优划分特征以及其对应的最优划分点
            GiniIndex = bestSplitGini  # 将当前的特征的最小基尼指数保存到GiniIndex变量之中

        # 对离散型特征进行处理, 非数字信息可以认为是离散的,Eg:'是' or '否'
        else:
            uniqueVals = set(featList)  # 获取该特征值可能的取值, 设置为集和避免重复
            newGiniIndex = 0.0  # 初始化基尼指数
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)  # 划分数据集, 取出该特征取值为value的所有样本
                prob = len(subDataSet) / float(len(dataSet))  # 计算子集占总数据集的概率
                newGiniIndex += prob * calGini(subDataSet)  # 计算基尼指数
            GiniIndex = newGiniIndex  # 将当前的特征的最小基尼指数保存到GiniIndex变量之中

        if GiniIndex < bestGiniIndex:
            bestGiniIndex = GiniIndex  # 更新的最小基尼指数
            bestFeature = i  # 并记录当前最佳划分特征

    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    # 二值化处理, 将数据划分为0,1, 简化数据, 方便计算, 但有可能损失信息
    #print('i am there')
    #print(is_number(dataSet[0][bestFeature]))
    if is_number(dataSet[0][bestFeature]):
        print('i am here')
        bestSplitValue = bestSplitDict[labels[bestFeature]]  # 获取最优划分特征的最优划分点
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)  # 更新该特征值
        labels_full[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)  # 更新该特征值
        for i in range(shape(dataSet)[0]):  # 遍历所有样本, 对样本中的该特征进行二值化处理
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1  # 小于等于划分点的值记为1
                dataSet_full[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0  # 大于划分点的值记为0
                dataSet_full[i][bestFeature] = 0
    return bestFeature

# 返回每个结点出现次数最多的类别,作为其分类名称
def majorClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #降序排序，可以指定reverse = true
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
# 网上的代码, 不好用
# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)
'''

#主程序，递归产生决策树
def createTree(dataSet,labels,dataSet_full, labels_full):
    # print(labels)
    classList = [example[-1] for example in dataSet]  # 获取最后一维的分类信息, 存储到列表中
    if classList.count(classList[0]) == len(classList):  # 如果都是所有的实例都是同树一类, 以该类作为结点的标记,返回单结点树
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果数据集的实例只有一个维度(也就是没有特征), 返回单结点树,以出现次数最多的那个类的值作为该点的类标记
        return majorClass(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet,labels,dataSetCopy,labels_full)  # 选择最好的划分特征的下标
    #print(dataSetCopy)
    #print(dataSet)
    tempLabel = copy.deepcopy(labels)
    bestFeatLabel = labels[bestFeat]  # 选择最好的划分特征
    if bestFeat != -1:  # 如果最佳特征为-1, 该特征不能作为分类特征, 则说明前面的特征已经用完, 这时可以结束, 返回树
        #print({bestFeatLabel:{}})
        myTree = {bestFeatLabel:{}}  # 以该特征构建结点
        del(labels[bestFeat])  # 将该特征从特征列表之中移除
        featValues = [example[bestFeat] for example in dataSet]  # 从数据集中获取该特征
        uniqueVals = set(featValues)  # 获取该特征所有可能的取值,放到集合之中, 二值化数据的优势在这里, 无论是连续还是离散数据, 二值花之后都可以用离散的方法处理
        for value in uniqueVals:  # 以该特征可能的取值将数据集划分为若干子集, 构建子结点, 递归调用, 在子结点基础上,构建子树
            subLabels = labels[:]  # 获取划分之后的标签
            subDataSet = splitDataSet(dataSet, bestFeat, value)  # 划分数据集, 将已经用过一次分类的特征从数据集中剔除, 一面重复
            myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels, dataSet_full, labelsCopy)  # 在子结点基础上,构建子树
        return myTree

#主程序，递归产生决策树, 这里包含了预剪枝
def createTreeProPruning(dataSet,labels,dataSet_full, labels_full,dataTest):
    # print(labels)
    classList = [example[-1] for example in dataSet]  # 获取最后一维的分类信息, 存储到列表中
    if classList.count(classList[0]) == len(classList):  # 如果都是所有的实例都是同树一类, 以该类作为结点的标记,返回单结点树
        return classList[0]
    if len(dataSet[0]) == 1:  # 如果数据集的实例只有一个维度(也就是没有特征), 返回单结点树,以出现次数最多的那个类的值作为该点的类标记
        return majorClass(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet,labels,dataSetCopy,labels_full)  # 选择最好的划分特征的下标
    #print(dataSetCopy)
    #print(dataSet)
    tempLabel = copy.deepcopy(labels)  # 复制一份标签, 用于预分类
    bestFeatLabel = labels[bestFeat]  # 选择最好的划分特征
    if bestFeat != -1:  # 如果最佳特征为-1, 该特征不能作为分类特征, 则说明前面的特征已经用完, 这时可以结束, 返回树
        #print({bestFeatLabel:{}})
        myTree = {bestFeatLabel:{}}  # 以该特征构建结点
        del(labels[bestFeat])  # 将该特征从特征列表之中移除
        featValues = [example[bestFeat] for example in dataSet]  # 从数据集中获取该特征
        uniqueVals = set(featValues)  # 获取该特征所有可能的取值,放到集合之中, 二值化数据的优势在这里, 无论是连续还是离散数据, 二值花之后都可以用离散的方法处理
        for value in uniqueVals:  # 以该特征可能的取值将数据集划分为若干子集, 构建子结点, 递归调用, 在子结点基础上,构建子树
            subLabels = labels[:]  # 获取划分之后的标签
            subDataSet = splitDataSet(dataSet, bestFeat, value)  # 划分数据集, 将已经用过一次分类的特征从数据集中剔除, 一面重复
            myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels, dataSet_full, labelsCopy)  # 在子结点基础上,构建子树
        # 进行测试，若划分没有提高准确率，则不进行划分，返回以出现次数最多的那个类的值作为该点的类标记
        if testing(myTree, dataTest, tempLabel) < testingMajor(majorClass(classList), dataTest):
            return myTree
        return majorClass(classList)

'''
# 网上的代码不好使
# 主程序，递归产生决策树
def createTree1(dataSet, labels, data_full, labels_full):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])
    # 针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree1(splitDataSet(dataSet, bestFeat, value), subLabels, data_full, labels_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree
'''

#由于在Tree中，连续值特征的名称以及改为了  feature<=value的形式
#因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
# 决策树的分类函数, 找到目标结点所对应的结点,返回节点的分类标签
def classify(inputTree,featLabels,testVec):  # 传入的数据为dict类型
    firstStr =list(inputTree.keys())[0]  # 第一个结点的特征
    if '<=' in firstStr:  # 如果这个结点标签是连续型数据的标签, 按处理连续型数据的方式进行处理
        featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])  # 获取这个连续型特征的最优切分点的值
        featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]  # 获取这个特征值
        secondDict=inputTree[firstStr]  # 建一个字典, 获取第一个结点的子树
        featIndex = featLabels.index(featkey)  # 获取这个特征值对在标签列表之中对应的下标
        if testVec[featIndex]<=featvalue:  # 如果测试样本特征值小于最优切分点的值, 记录唯1
            judge=1
        else:  # 否则记录为0
            judge=0
        for key in secondDict.keys():  # 获取该结点对应的分类条件, 并且遍历 Eg：是 或者 否
            if judge == int(key):  # 这是是二分叉, 判断是往左走还是往右走
                if type(secondDict[key]).__name__=='dict':  # 若果子结点不是叶结点, 还有子树, 继续往下搜索
                    classLabel = classify(secondDict[key],featLabels,testVec)  # 递归往下搜索
                else:  # 若果这个子结点是叶结点, 则该结点的标签即为测试集的标签
                    classLabel = secondDict[key]
    else:  # 如果这个结点标签是离散型数据的标签, 按处理离散型数据的方式进行处理
        secondDict=inputTree[firstStr]  # 建一个字典, 获取第一个结点的子树
        featIndex=featLabels.index(firstStr)  # 获取这个特征值对在标签列表之中对应的下标
        for key in secondDict.keys():  # 获取该结点对应的分类条件, 并且遍历 Eg：是 或者 否
            if testVec[featIndex]==key:   # 这是是二分叉, 判断是往左走还是往右走
                if type(secondDict[key]).__name__=='dict':  # 若果子结点不是叶结点, 还有子树, 继续往下搜索
                    classLabel=classify(secondDict[key],featLabels,testVec)  # 递归往下搜索
                else:   # 若果这个子结点是叶结点, 则该结点的标签即为测试集的标签
                    classLabel=secondDict[key]
    return classLabel

# 测试该树的对测试数据的分类的错误率
def testing(myTree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    print('myTree error ratings is %d' % error)
    return float(error)

# 参数major是一个标签值, 参数data_test是一个测试数据集合
# 函数目的测试测试集中有多少个样本未被正确分类
def testingMajor(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    print('major %d' % error)
    return float(error)



#后剪枝
# 以下是进行剪枝数处理
# 思想：用与当前树一样的结构对测试集进行处理, 生成一结构一样的树(称为测试树或者测试集),
#      1、计算测试树与当前树叶结点的误差
#      2、计算测试树与当前树的叶结点的父结点的误差
#      3、如果 1 > 2, 对叶结点进行剪枝, 以叶结点的父结点作为最新的叶结点; 否则, 不做改变
#      4、重复1～3, 继续往根结点进行回溯,直到遇到根结点后, 停止剪枝
def postPruningTree(inputTree,dataSet,data_test,labels):
    firstStr = list(inputTree.keys())[0]  # 第一个结点的特征
    secondDict=inputTree[firstStr]  # 建一个字典, 获取第一个结点的子树
    classList=[example[-1] for example in dataSet]  # 获取最后一维的分类信息, 存储到列表中
    featkey=copy.deepcopy(firstStr)  # 复制一份第一结点的特征
    if '<=' in firstStr:  # 如果这个结点标签是连续型数据的标签, 按处理连续型数据的方式进行处理
        featkey=re.compile("(.+<=)").search(firstStr).group()[:-2]  # 获取这个特征值
        featvalue=float(re.compile("(<=.+)").search(firstStr).group()[2:])  # 获取这个连续型特征的最优切分点的值
    labelIndex=labels.index(featkey)   # 获取这个特征值对在标签列表之中对应的下标
    temp_labels=copy.deepcopy(labels)  # 复制一份标签
    del(labels[labelIndex])  # 将该特征从特征列表之中移除
    for key in secondDict.keys():  # 获取该结点对应的分类条件, 并且遍历 Eg：是 或者 否
                                   # 这是是二分叉, 判断是往左走还是往右走
        if type(secondDict[key]).__name__=='dict':  # 若该子结点不是叶结点, 还有子树, 继续往下搜索
            if type(dataSet[0][labelIndex]).__name__=='str':  # 如果是离散性数据, 按离散型的方式处理
                # 在在子树的基础上继续往下搜索
                inputTree[firstStr][key]=postPruningTree(secondDict[key],  # 子树
                                        splitDataSet(dataSet,labelIndex,key),  # 这是是二分叉, 划分数据集
                                        splitDataSet(data_test,labelIndex,key),  # 用与当前树一样的结构对测试集进行划分
                                        copy.deepcopy(labels))  # 标签
            else:  # 如果是连续数据, 按离散型的方式处理
                # 在在子树的基础上继续往下搜索
                inputTree[firstStr][key]=postPruningTree(secondDict[key],  # 子树
                                        splitContinuousDataSet(dataSet,labelIndex,featvalue,key),  # 这是是二分叉, 划分数据集
                                        splitContinuousDataSet(data_test,labelIndex,featvalue,key),  # 用与当前树一样的结构对测试集进行划分
                                        copy.deepcopy(labels))  # 标签
    # 进行测试,若划分没有提高准确率,则进行剪枝,返回以出现次数最多的那个类的值作为该点的类标记
    if testing(inputTree,data_test,temp_labels)<=testingMajor(majorClass(classList),data_test): #
        return inputTree
    return majorClass(classList)


dataSet, labels = readDataFromFile('CARTDataSet(回归树).txt')
labelsCopy = labels.copy()
dataSetCopy = dataSet.copy()
dataSetTest, labelsTest = readDataFromFile('CARTDataSet(回归树)测试集.txt')
mytree = createTree(dataSet, labels, dataSetCopy, labelsCopy)
for item in dataSetTest:
    print(classify(mytree,labelsCopy,item))
createPlot(mytree)