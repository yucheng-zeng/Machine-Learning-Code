import numpy as np

# 输入格式如下：
# A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])
# B = np.array([[.5,.5],[.4,.6],[.7,.3]])
# Pi = np.array([[.2,.4,.4]])
# O = np.array([[1,2,1]])

# 应用ndarray在数组之间进行相互运算时，一定要确保数组维数相同！
# 比如：
# In[93]:m = np.array([1,2,3,4])
# In[94]:m
# Out[94]: array([1, 2, 3, 4])
# In[95]:m.shape
# Out[95]: (4,)
# 这里表示的是一维数组
# In[96]:m = np.array([[1,2,3,4]])
# In[97]:m
# Out[97]: array([[1, 2, 3, 4]])
# In[98]:m.shape
# Out[98]: (1, 4)
# 而这里表示的就是二维数组
# 注意In[93]和In[96]的区别,多一对中括号！！

# N = A.shape[0]为数组A的行数， H = O.shape[1]为数组O的列数
# 在下列各函数中，alpha数组和beta数组均为N*H二维数组，也就是横向坐标是时间，纵向是状态


# 前向算法
def ForwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数, 代表有多少个步
    M = A.shape[1]  # 数组A的列数, 代表每步有多少个选择
    H = O.shape[1]  # 数组O的列数, 代表有多少次抽取

    sum_alpha_1 = np.zeros((M, N))  # 初始化前向概率矩阵
    alpha = np.zeros((N, H))  # 初始化前向概率矩阵
    r = np.zeros((1, N))  #
    alpha_1 = np.multiply(Pi[0, :], B[:, O[0, 0] - 1])  # 初始化前向概率矩阵
    # alpha_1是一维数组，在使用np.multiply的时候需要升级到二维数组
    alpha[:, 0] = np.array(alpha_1).reshape(1, N)  # 将初始值赋值给前向概率矩阵

    for h in range(1, H):  # 从第二步开始, 向前迭代, 直到最后一次抽取
        for i in range(N):  # 遍历每一步
            for j in range(M):  # 遍历每一种选择
                sum_alpha_1[i, j] = alpha[j, h - 1] * A[j, i]  # 计算第i步的值
            r = sum_alpha_1.sum(1).reshape(1, N)  # 将值赋值前向概率矩阵
            alpha[i, h] = r[0, i] * B[i, O[0, h] - 1]  # 计算at+1(i)
    # print("alpha矩阵: \n %r" % alpha)
    # 计算P(O|lambda)
    p = alpha.sum(0).reshape(1, H)
    P = p[0, H - 1]
    # print("观测概率: \n %r" % P)
    # return alpha
    return alpha, P


def BackwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数, 代表有多少个步
    M = A.shape[1]  # 数组A的列数, 代表每步模型有多少个选择
    H = O.shape[1]  # 数组O的列数, 代表有多少次抽取

    # beta = np.zeros((N,H))
    sum_beta = np.zeros((1, N))  # 初始化后向概率矩阵
    beta = np.zeros((N, H))  # 初始化前向概率矩阵
    beta[:, H - 1] = 1  # 初始化最后一步的值
    p_beta = np.zeros((1, N))  #

    for h in range(H - 1, 0, -1):  # 从倒数第二步开始, 向后迭代, 直到第一次抽取
        for i in range(N):  # 遍历每一步
            for j in range(M):  # 遍历每一种选择
                sum_beta[0, j] = A[i, j] * B[j, O[0, h] - 1] * beta[j, h]  # 计算第i步的值
            beta[i, h - 1] = sum_beta.sum(1)  # 将值赋值给后巷概率矩阵
    # print("beta矩阵: \n %r" % beta)
    for i in range(N):  #
        p_beta[0, i] = Pi[0, i] * B[i, O[0, 0] - 1] * beta[i, 0]
    p = p_beta.sum(1).reshape(1, 1)  # 计算P(o|lambda)
    # print("观测概率: \n %r" % p[0,0])
    return beta, p[0, 0]


def FBAlgoAppli(A, B, Pi, O, I):
    # 计算在观测序列和模型参数确定的情况下，某一个隐含状态对应相应的观测状态的概率
    # 例题参考李航《统计学习方法》P189习题10.2
    # 输入格式：
    # I为二维数组，存放所求概率P(it = qi,O|lambda)中it和qi的角标t和i，即P=[t,i]
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    p = alpha[I[0, 1] - 1, I[0, 0] - 1] * beta[I[0, 1] - 1, I[0, 0] - 1] / p1
    return p

# 公式10.24, 获取gamma值
def GetGamma(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    H = O.shape[1]  # 数组O的列数
    Gamma = np.zeros((N, H))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H):
        for i in range(N):
            Gamma[i, h] = alpha[i, h] * beta[i, h] / p1
    return Gamma

# 公式10.26, 获取sigma值
def GetSigma(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数
    Xi = np.zeros((H - 1, N, M))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H - 1):
        for i in range(N):
            for j in range(M):
                Xi[h, i, j] = alpha[i, h] * A[i, j] * B[j, O[0, h + 1] - 1] * beta[j, h + 1] / p1
    # print("Xi矩阵: \n %r" % Xi)
    return Xi


# 单次迭代的BaumWelch算法, 用于生成模型
def BaumWelchAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数, 代表有多少个步
    M = A.shape[1]  # 数组A的列数, 代表每步模型有多少个选择
    Y = B.shape[1]  # 数组B的列数, 代表右多少类抽取结果
    H = O.shape[1]  # 数组O的列数, 代表有多少次抽取

    c = 0
    Gamma = GetGamma(A, B, Pi, O)  # 获取gamma值
    Xi = GetSigma(A, B, Pi, O)  # 获取sigma
    Xi_1 = Xi.sum(0)
    a = np.zeros((N, M))  # 初始化转移概率矩阵
    b = np.zeros((M, Y))  # 初始化观测概率矩阵
    pi = np.zeros((1, N))  # 初始化 初始概率状态

    # 公式 10.39
    a_1 = np.subtract(Gamma.sum(1), Gamma[:, H - 1]).reshape(1, N)  #
    for i in range(N):
        for j in range(M):
            a[i, j] = Xi_1[i, j] / a_1[0, i]

    # 公式 10.40
    for y in range(Y):
        for j in range(M):
            for h in range(H):
                if O[0, h] - 1 == y:
                    c = c + Gamma[j, h]
            gamma = Gamma.sum(1).reshape(1, N)
            b[j, y] = c / gamma[0, j]
            c = 0

    # 公式 10.41
    for i in range(N):
        pi[0, i] = Gamma[i, 0]
    # print(pi)
    return a, b, pi

# 计算迭代次数为n的BaumWelch算法, 用于生成模型
def BaumWelchAlgo_n(A, B, Pi, O, n):
    for i in range(n):
        A, B, Pi = BaumWelchAlgo(A, B, Pi, O)
    return A, B, Pi


# 维特比算法
def viterbi(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数, 代表有多少个步
    M = A.shape[1]  # 数组A的列数, 代表每步模型有多少个选择
    H = O.shape[1]  # 数组O的列数, 代表有多少次抽取
    Delta = np.zeros((M, H))  # 初始Delta
    Psi = np.zeros((M, H))  # 初始哈Psi
    Delta_1 = np.zeros((N, 1))  # 初始化
    I = np.zeros((1, H))  # 初始化最优路径

    # 公式10.44,  计算P(it+1=i,it=,...,...i1,ot+1,...,o1|lambda)
    for i in range(N):
        Delta[i, 0] = Pi[0, i] * B[i, O[0, 0] - 1]

    # 公式10.45, 公式10.46
    for h in range(1, H):
        for j in range(M):
            for i in range(N):
                Delta_1[i, 0] = Delta[i, h - 1] * A[i, j] * B[j, O[0, h] - 1]
            Delta[j, h] = np.amax(Delta_1)
            Psi[j, h] = np.argmax(Delta_1) + 1
    #print("Delta矩阵: \n %r" % Delta)
    print("Psi矩阵: \n %r" % Psi)
    P_best = np.amax(Delta[:, H - 1])
    psi = np.argmax(Delta[:, H - 1])
    I[0, H - 1] = psi + 1
    for h in range(H - 1, 0, -1):
        index = int(I[0, h]-1)
        I[0, h - 1] = Psi[index, h]
    print("最优路径概率: \n %r" % P_best)
    print("最优路径: \n %r" % I)


if __name__=='__main__':

    # 状态转移矩阵, aij代表第i+1次, 转移到第j+1个状态的概率
    A = np.array([[.5,.2,.3],
                  [.3,.5,.2],
                  [.2,.3,.5]])

    # 观测概率矩阵, bij代表第i+1次, 抽到的第j+1类样本的该概率
    B = np.array([[.5,.5],
                  [.4,.6],
                  [.7,.3]])

    Pi = np.array([[.2,.4,.4]])  # 初始状态概率向量
    O = np.array([[1,2,1]])  # 观测序列

    #print(ForwardAlgo(A,B,Pi,O))
    #print(BackwardAlgo(A,B,Pi,O))
    #print(GetGamma(A,B,Pi,O))
    #print(BaumWelchAlgo_n(A,B,Pi,O,10))
    print(viterbi(A,B,Pi,O))
