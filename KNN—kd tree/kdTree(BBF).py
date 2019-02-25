import numpy as np


class Node:  # 结点
    def __init__(self, data, leftChild=None, rightChild=None):
        self.data = data  # 结点自身数据
        self.leftChild = leftChild  # 结点左侧数据集
        self.rightChild = rightChild  # 结点右侧数据集合


class KdTree:  # kd树
    def __init__(self):
        self.kdTree = None

    def create(self, dataSet, depth):  # 创建kd树，返回根结点,通过根结点可以找到整棵树
        if (len(dataSet) > 0):
            row, column = np.shape(dataSet)  # 求出样本行，列
            midIndex = row//2  # 中间数的索引位置
            axis = depth % column  # 判断以哪个轴划分数据
            sortedDataSet = dataSet[:]  # 为不破坏原数据,将原数据复制一份
            sortedDataSet.sort(key=lambda x:x[axis])  # 按维度为axis对数据进行由小到大的排序
            node = Node(sortedDataSet[midIndex])  # 将节点数据域设置为中位数
            # print sortedDataSet[midIndex]
            leftDataSet = sortedDataSet[: midIndex]  # 将中位数的左边数据分离出来
            rightDataSet = sortedDataSet[midIndex + 1:]  # 将中位数右边的数据分离出来
            print(leftDataSet)
            print(rightDataSet)
            # 递归创建树
            node.leftChild = self.create(leftDataSet, depth + 1)  # 将中位数左边样本传入来递归创建子树
            node.rightChild = self.create(rightDataSet, depth + 1)   # 将中位数右边样本传入来递归创建子树
            return node
        else:
            return None

    def preOrder(self, node, depth=0):  # 前序遍历，递归打印书信息
        if node != None:

            print("第%s层%s"%(depth,node.data))
            depth += 1
            self.preOrder(node.leftChild,depth)
            self.preOrder(node.rightChild,depth)

    def search(self, tree, target):  # BBF算法,用kd树的最近邻搜索
        self.nearestPoint = None  # 保存最近的点
        self.nearestValue = 0  # 保存目标点与最近点的欧氏距离

        # 递归搜索,直到找到包含目标点的叶结点（最小超矩形区域）
        def travel(node, depth=0):  # 递归搜索
            if node != None:  # 递归终止条件
                n = len(target)  # 目标点的特征数
                axis = depth % n  # 计算轴
                if target[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
                    travel(node.leftChild, depth + 1)
                else:
                    travel(node.rightChild, depth + 1)

                # 以下是递归完毕后,现在叶结点位置处,往父结点方向回朔,找到不断更新当前最近点,找到最近邻点
                distNodeAndTarget = self.distance(target, node.data)  # 计算目标和节点的欧氏距离
                if (self.nearestPoint == None):  # 确定当前点，更新最近的点和最近的值
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndTarget
                elif (distNodeAndTarget < self.nearestValue):  # 存在更近的点, 更新最近的点和最近的值
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndTarget

                #print(node.data, depth, self.nearestValue, node.data[axis], target[axis])

                if (abs(target[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去父结点的子节点的区域去找（圆的判断）
                                                                                # 用哈密顿距离来判断是否是相交,轴是一致的
                    #print('abs:'+str(abs(target[axis] - node.data[axis])))
                    if target[axis] < node.data[axis]:
                        travel(node.rightChild, depth + 1)
                    else:
                        travel(node.leftChild, depth + 1)

        travel(tree)
        return self.nearestPoint

    def distance(self, x1, x2):  # 欧式距离的计算
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5


if __name__ == '__main__':
    dataSet = [[2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [8, 1],
               [7, 2]]
    target = [4, 4]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)
    kdtree.preOrder(tree)
    print(kdtree.search(tree, target))