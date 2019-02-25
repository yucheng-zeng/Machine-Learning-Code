from scipy.spatial.distance import pdist, squareform
from scipy import exp
from numpy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 线性核化降维
def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA 实现.
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_samples, n_features]

    gamma: float
      RBF核的调优参数

    n_components: int
      要返回的主要组件的数量
    Returns
    ------------
     X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
       Projected dataset
    """
    # 计算成对的欧几里得距离。
    # 在MxN维数据集中
    sq_dists = pdist(X, 'sqeuclidean')

    # 将成对距离转换成方阵。
    mat_sq_dists = squareform(sq_dists)

    # 计算对称核矩阵。
    K = exp(-gamma * mat_sq_dists)

    # 中心核矩阵.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # 从中心核矩阵得到特征对
    # numpy.eigh 按顺序返回它们
    eigvals, eigvecs = eigh(K)

    # 收集顶级k特征向量(投影样本)
    X_pc = np.column_stack((eigvecs[:, -i]for i in range(1, n_components + 1)))

    return X_pc


def origin_dataSet():
    X, y = make_moons(n_samples=100, random_state=123)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

    plt.tight_layout()
    plt.show()




def PCA():
    from sklearn.decomposition import PCA
    X, y = make_moons(n_samples=100, random_state=123)
    print(X)
    scikit_pca = PCA(n_components=2)
    X_spca = scikit_pca.fit_transform(X)
    print(X_spca)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

    ax[0].scatter(X_spca[y == 0, 0], X_spca[y == 0, 1],color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_spca[y == 1, 0], X_spca[y == 1, 1],color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_spca[y == 0, 0], np.zeros((50, 1)) + 0.02,color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_spca[y == 1, 0], np.zeros((50, 1)) - 0.02,color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')

    plt.tight_layout()
    # plt.savefig('./figures/half_moon_2.png', dpi=300)
    plt.show()

def KPCA1():
    from matplotlib.ticker import FormatStrFormatter
    X, y = make_moons(n_samples=100, random_state=123)

    X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

    fig, ax = plt.subplots(nrows=1,ncols=2, figsize=(7,3))
    ax[0].scatter(X_kpca[y==0, 0], X_kpca[y==0, 1],color='red', marker='^', alpha=0.5)
    ax[0].scatter(X_kpca[y==1, 0], X_kpca[y==1, 1],color='blue', marker='o', alpha=0.5)

    ax[1].scatter(X_kpca[y==0, 0], np.zeros((50,1))+0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(X_kpca[y==1, 0], np.zeros((50,1))-0.02,color='blue', marker='o', alpha=0.5)

    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    plt.tight_layout()
    plt.show()


def KPCA2():
    from sklearn.decomposition import KernelPCA

    X, y = make_moons(n_samples=100, random_state=123)
    # n_components 表示降维之后的维度数,kernel表示运用数目核函数, gamma表示参数
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_skernpca = scikit_kpca.fit_transform(X)  # 生成核化之后的X, 在此基础上运用PCA

    plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],color='red', marker='^', alpha=0.5)
    plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],color='blue', marker='o', alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    KPCA2()