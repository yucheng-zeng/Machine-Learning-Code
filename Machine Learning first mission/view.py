import common_function as function
import numpy as np
import matplotlib.pyplot as plt
import gradient_descent as gd
import conjugate_gradient as cg
import least_squares as ls

if __name__ == '__main__':
    cf = function.config(0, 2*np.pi, 100, 5, 1e-6, 20000, 1e-5, 0.1)
    x, y = function.create_data(cf.x0, cf.x1, cf.n)
    matX, matY = function.create_mat(x, y, cf.k)

    # 最小二乘法,无正则项
    matA1 = ls.fitting(matX, matY, cf)

    # 最小二乘法，有正则项
    matA2 = ls.fittingRegular(matX, matY, cf)

    # 梯度下降法，无正则项
    cf.theta = 0
    matA3 = gd.SGD(matX, matY, cf)

    # 梯度下降法，有正则项
    cf.theta = 0.3
    matA4 = gd.SGD(matX, matY, cf)

    # 共轭梯度法
    matA5 = cg.CG(matX, matY, cf)

    plt.plot(x, y, color='b', linestyle='', marker='.')
    plt.plot(x, matX.dot(matA1), color='R', linestyle='-', marker='', label='least squares')

    plt.plot(x, matX.dot(matA2), color='b', linestyle='-', marker='', label='least squares with regular')

    plt.plot(x, matX.dot(matA3), color='c', linestyle='-', marker='', label='gradient descent')

    plt.plot(x, matX.dot(matA4), color='m', linestyle='-', marker='', label='gradient descent with regular')

    plt.plot(x, matX.dot(matA5), color='g', linestyle='-', marker='', label='conjugate gradient')
    plt.legend(loc='upper right')
    print('least squares                 =', function.RMSE(matX,matY,matA1))
    print('least squares with regular    =', function.RMSE(matX, matY, matA2))
    print('gradient descent              =', function.RMSE(matX, matY, matA3))
    print('gradient descent with regular =', function.RMSE(matX, matY, matA4))
    print('conjugate gradient            =', function.RMSE(matX, matY, matA5))
    plt.show()

