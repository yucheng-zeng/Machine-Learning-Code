import numpy as np
import matplotlib.pyplot as plt
import common_function as function

def mini_banch_SGD(matX, matY, cf):
    '''
    function:批量随机梯度下降
    :param matX:
    :param matY:
    :return:
    '''
    count = 0  # 当前迭代次数
    matA = np.random.randn(cf.k)
    old_matA = matA
    # 确认最小梯度
    if cf.n/10 > 1:
        least_banch = int(cf.n/10)
    else:
        least_banch = 1
    while count < cf.loop_max:
        sum_m = np.zeros(cf.k)  # 记录一批数据的梯度差
        count += 1
        chosen = np.random.randint(0, cf.n, least_banch)  # 生成随机数
        for i in chosen:
            # 计算梯度
            diff = matX[i].dot(matX[i].dot(matA.T) - matY[i])
            sum_m = sum_m + diff  # 计算梯度和
        sum_m = sum_m/len(chosen)

        if np.linalg.norm(sum_m) < cf.epsilon:
            break
        # 带有正则化的loss
        matA = matA*(1-cf.theta*cf.alpha/len(chosen)) - cf.alpha*sum_m
        # 提前结束
        if np.linalg.norm(matA - old_matA) < cf.epsilon:
            break
        old_matA = matA
        print('count=', count, 'loss=', function.RMSE(matX, matY, matA))
    return matA

def SGD(matX, matY, cf):
    count = 0  # 当前迭代次数
    matA = np.zeros(cf.k)
    old_matA = matA
    while count < cf.loop_max:
        count += 1
        for i in range(cf.n):
            # 计算梯度
            diff = matX[i].dot(matX[i].dot(matA.T) - matY[i])
            # 带有正则化的loss
            matA = matA * (1 - cf.theta * cf.alpha) - cf.alpha * diff

        # 提前结束
        if np.linalg.norm(matA-old_matA) < cf.epsilon:
            break
        old_matA = matA
        print('count=',count,'loss=',function.RMSE(matX,matY,matA))
    return matA

if __name__ == '__main__':
    cf = function.config(0, 2*np.pi, 100, 5, 1e-6, 500000, 1e-5, 0.1)
    x, y = function.create_data(cf.x0, cf.x1, cf.n)
    matX, matY = function.create_mat(x, y, cf.k)
    matA = SGD(matX, matY, cf)
    plt.title('gradient descent with regular')
    function.show(x, y, matX, matA)

