from numpy import *

def createDataSet():
    dataSet = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]  # 创建数据集
    feature = len(dataSet[0])  # 每个样本点的维度
    return dataSet, feature

def createTree(dataSet, feature, layer = 0):
    length = len(dataSet)  # 样本点的个树
    dataSetCopy = dataSet[:]  # 复制一份数据,以免破坏原本的数据
    featureNum = layer % feature  # 确定当前按哪个维度进行划分
    dataSetCopy.sort(key=lambda x:x[featureNum])  # 对第featureNum个维度进行排序
    layer += 1
    if length == 0:  # 若数据集没有数据,退出
        return None
    elif length == 1:  # 若数据集里只有一个样本点,当前样本点即为根结点，即为整棵树
        return {'Value':dataSet[0], 'Layer':layer, 'feature':featureNum, 'Left':None, 'Right': None}
    elif length != 1:
        midNum = length//2  # 找到中位数的位置索引
        dataSetLeft = dataSetCopy[:midNum]  # 将中位数的左边数据分离出来
        dataSetRight = dataSetCopy[midNum+1:]  # 将中位数右边的数据分离出来
        return {'Value':dataSetCopy[midNum], 'Layer':layer, 'feature':featureNum,
                'Left':createTree(dataSetLeft,feature,layer),  # 将中位数左边样本传入来递归创建子树
                'Right': createTree(dataSetRight,feature,layer)  # 将中位数右边样本传入来递归创建子树
               }

def calDistance(sourcePoint,targetPoint):  # 计算欧氏距离
    length = len(targetPoint)  # 目标点的维度数
    distance = 0.0
    for i in range(length):
        distance += (sourcePoint[i] - targetPoint[i])**2
    distance = sqrt(distance)
    return distance

# 用深度优先搜索算法进行搜索, 找到包含目标点的叶结点（最小超矩形区域）
# 返回包含目标点的叶结点值, 以及其的祖先结点(即包含目标点的叶结点的路径)
def dfs(kdTree, target,tracklist = []):
    tracklistCopy = tracklist[:]  # 获取当前树的树的副本， 不要改变原来的树结构
    if not kdTree:  # 如果树为空, 结点返回空
        return None, tracklistCopy
    elif not (kdTree['Left'] or kdTree['Right']):  # 如若当前的结点不为空, 但其左子结点和右子结点的都为空, 即到达末尾
        tracklistCopy.append(kdTree['Value'])  # 列表增加当前树
        #print('返回值')
        #print(kdTree['Value'])
        return kdTree['Value'], tracklistCopy  # 返回当前父结点值, 返回当前树
    elif kdTree['Left'] or kdTree['Right']:  # 如若当前的结点不为空, 且其左子结点或者右子结点的不为空
        pointValue = kdTree['Value']  # 获取当前结点
        feature = kdTree['feature']  # 获取当前结点分类依据的维度
        tracklistCopy.append(pointValue)
        # 递归搜索,直到找到包含目标点的叶结点（最小超矩形区域）
        if target[feature] <= pointValue[feature]:   # 若小于等于,往左搜
            return dfs(kdTree['Left'], target, tracklistCopy)
        elif target[feature] > pointValue[feature]:  # 若大于,往右搜索
            return dfs(kdTree['Right'], target, tracklistCopy)

# 在树中找结点, 返回该结点对应的子树
def findPoint(Tree, value):
    if Tree !=None and Tree['Value'] == value:
        return Tree
    else:
        if Tree['Left'] != None:
            return findPoint(Tree['Left'], value) or findPoint(Tree['Right'], value)


# 找到包含目标点的叶结点值,再层层往上回溯,找最近邻点
def kdTreeSearch(kdTree, tracklist, target, usedPoint=[], minDistance=float('inf'), minDistancePoint=None):
    tracklistCopy = tracklist[:]  # 复制一份包含目标点的叶结点的路径
    usedPointCopy = usedPoint[:]  # 复制一份遍历过的点

    if not minDistancePoint:  # 如果没有当前最近点, 则路径中的最后一个结点为当前最近点(初始化的过程)
        minDistancePoint = tracklistCopy[-1]

    if len(tracklistCopy) == 1:  # 如果路径中就是只有一个点, 则这个点就是最近邻点
        return minDistancePoint
    else:
        point = findPoint(kdTree, tracklist[-1])  # 找到包含目标点的叶结点的信息

        if calDistance(point['Value'], target) < minDistance:  # 计算结点与目标点的欧氏距离, 如果比当前最近点的距离小
                                                               # 更新当前点为当前最近点, 更新当前最近距离
            minDistance = calDistance(point['Value'], target)
            minDistancePoint = point['Value']
        fatherPoint = findPoint(kdTree, tracklistCopy[-2])  # 找到上一层父结点的信息
        fatherPointval = fatherPoint['Value']  # 父结点的坐标值
        fatherPointfea = fatherPoint['feature']  # 父结点的分类依据的维度数

        if calDistance(fatherPoint['Value'], target) < minDistance:  # 计算父结点与目标点的距离,如果比当前最近点的距离小
                                                                     # 更新当前点为当前最近点, 更新当前最近距离
            minDistance = calDistance(fatherPoint['Value'], target)
            minDistancePoint = fatherPoint['Value']

        # 计算找到包含目标点的叶结点是左结点还是右结点, 把另外一个结点找出来(以下称作另外一个结点)
        if point == fatherPoint['Left']:
            anotherPoint = fatherPoint['Right']
        elif point == fatherPoint['Right']:
            anotherPoint = fatherPoint['Left']

        if (anotherPoint == None or anotherPoint['Value'] in usedPointCopy or
                abs(fatherPointval[fatherPointfea] - target[fatherPointfea]) > minDistance):
            # 如果另外一个结点为空, 或者已经遍历过了, 或者另外一个结点的与目标结点的距离比最近距离大
            # 则包含目标点的叶结点这一层已经搜索完毕
            usedPoint = tracklistCopy.pop()  # 将路径上最后的一个结点弹出
            usedPointCopy.append(usedPoint)  # 将其标记为已经遍过的结点
            # 递归, 以父结点（当前最近点）为根结点继续往上回溯
            return kdTreeSearch(kdTree, tracklistCopy, target, usedPointCopy, minDistance, minDistancePoint)
        else:
            # 如果另外一个结点的距离比当前距离小, 则设另外一个结点为当前最近
            usedPoint = tracklistCopy.pop()  # 将路径上最后的一个结点弹出
            usedPointCopy.append(usedPoint)  # 将其标记为已经遍过的结点
            subvalue, subtrackList = dfs(anotherPoint, target)
            tracklistCopy.extend(subtrackList)  # 将当前最近点重新加入路径
            # 递归, 以当前最近点向上回溯
            return kdTreeSearch(kdTree, tracklistCopy, target, usedPointCopy, minDistance, minDistancePoint)


dataSet, feature = createDataSet()
kdTree = createTree(dataSet, feature)
#print(kdTree)
target = (4,4)
value, trackList = dfs(kdTree, target)
nnPoint = kdTreeSearch(kdTree,trackList, target)
print('离目标点%s的最近点：%s'%(str(target),str(nnPoint)))

# kd树搜索思想一样, 都是先找到包含目标点的叶结点值,再层层往上回溯,找最近邻点
# 该程序仍有不足之处,其中一点即为kd书的储存结构并不是特别的优秀(用字典嵌套存储非常不好),导致搜索算法太复杂了
# 每次在使用的时候也都需要使用查找函数去找到value对应的节点，才能获得节点的各种属性。这大大增加了时间复杂度。
