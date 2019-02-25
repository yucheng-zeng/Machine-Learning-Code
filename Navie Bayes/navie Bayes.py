
'''
玩具级别小程序, 拓展性不强
'''
def loadDataSet():
    x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    return x1, x2, Y


def prior_probability(Y):
    py1 = Y.count(1) / len(Y)
    py2 = 1 - py1
    return py1, py2

# 计算条件概率
def conditional_probability(xj, value, Y):
    xcount = 0
    _xcount = 0
    for i in range(len(Y)):
        if Y[i] == 1 and xj[i] == value:
            xcount += 1
        elif Y[i] == -1 and xj[i] == value:
            _xcount += 1
    # print('x=%s,Y=%d,p=%f' % (value, 1, xcount / Y.count(1)))
    # print('x=%s,Y=%d,p=%f' % (value, -1, _xcount / Y.count(-1)))
    pxy1 = xcount / Y.count(1)
    pxy2 = _xcount / Y.count(-1)
    return pxy1, pxy2


def classify(x1, x2, inX, Y):
    px1y1, px1y2 = conditional_probability(x1, int(inX[0]), Y)  # 传入实例特征一, 计算特征一的条件概率
    px2y1, px2y2 = conditional_probability(x2, inX[1], Y)  # 传入实例特征二, 计算特征二的条件概率
    py1, py2 = prior_probability(Y)  # 计算先验概率
    p1Vec = py1 * px1y1 * px2y1  # 贝叶斯分类器结果
    p2Vec = py2 * px1y2 * px2y2  # 贝叶斯分类器结果
    if p1Vec > p2Vec:  # 比较贝叶斯分类器的结果
        print('x=(%s,%s)被分为Y=1' % (inX[0], inX[1]))
    else:
        print('x=(%s,%s)被分为Y=-1' % (inX[0], inX[1]))


if __name__=='__main__':
    x1, x2, Y = loadDataSet()
    inX = [2, 'S']
    classify(x1, x2, inX, Y)