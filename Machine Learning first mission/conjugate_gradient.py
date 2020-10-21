import numpy as np
import common_function as function
import matplotlib.pyplot as plt

def CG(matX, matY, cf):
    matY = matX.T.dot(matY)
    matX = matX.T.dot(matX)
    matA = np.random.rand(cf.k)
    r = matY - matX.dot(matA)
    p = r
    for i in range(cf.k):
        alpha = (r.T.dot(r))/(p.T.dot(matX).dot(p))
        matA = matA + alpha*p
        old_r = r
        r = r - alpha*matX.dot(p)
        if np.linalg.norm(r) < cf.epsilon:
            break
        beta = (r.T.dot(r))/(old_r.T.dot(old_r))
        p = r + beta*p
    return matA


if __name__ == '__main__':
    cf = function.config(0, 2*np.pi, 100, 5, 3e-7, 150000, 1e-20, 0.01)
    x, y = function.create_data(cf.x0, cf.x1, cf.n)
    matX, matY = function.create_mat(x, y, cf.k)
    matA = CG(matX, matY, cf)
    err = function.RMSE(matX, matY, matA)
    plt.title('conjugate gradient')
    function.show(x, y, matX, matA)