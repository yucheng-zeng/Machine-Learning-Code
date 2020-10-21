import numpy as np
import common_function as function
import matplotlib.pyplot as plt

def fitting(matX,matY,cf):
    #计算X.T*X
    squareX = np.dot(matX.T, matX)
    invX = np.linalg.inv(squareX)
    MatA = (invX.dot(matX.T)).dot(matY)
    return MatA

# 加入正则化的最小二乘法
def fittingRegular(matX,matY,cf):
    #计算X.T*X+NRI
    squareX = np.dot(matX.T, matX) + cf.n*cf.theta*np.eye(cf.k, cf.k)
    invX = np.linalg.inv(squareX)
    MatA = (invX.dot(matX.T)).dot(matY)
    return MatA

if __name__ == '__main__':
    cf = function.config(-20, 20, 200, 25, 3e-7, 150000, 1e-20, 0.0)
    x,y = function.create_data(cf.x0, cf.x1, cf.n)
    #print(x,y)
    matX, matY = function.create_mat(x,y,cf.k)
    #print(matX)
    #print(matY)
    #print(matA)
    matA = fitting(matX,matY,cf)
    #print(matA)
    err = function.RMSE(matX,matY,matA)
    print(err)
    plt.title('Least Squares with Regular')
    function.show(x,y,matX, matA)
