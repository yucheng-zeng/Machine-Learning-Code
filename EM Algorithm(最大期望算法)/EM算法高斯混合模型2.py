import math
import copy
import numpy as np
import matplotlib.pyplot as plt

isdebug = False


# 指定k个高斯分布參数。这里指定k=2。注意2个高斯分布具有同样均方差Sigma，分别为Mu1,Mu2。
def ini_data(Sigma,Mu1,Mu2,k,N):
    global X   # 记录正态分布的点
    global Mu  # 记录整体模型的Mu
    global Expectations  # 记录整体期望值
    X = np.zeros((1,N))  # 创建一个1×N的矩阵, 存储模型点的取值
    Mu = np.random.random(2)  # 随机初始化整体模型的Mu包含两个元素
    Expectations = np.zeros((N,k))  # 初始化期望值
    for i in range(0,N):
        if np.random.random(1) > 0.5:  # 随机生成点, 两个子模型的生成的点数一致
            X[0, i] = np.random.normal()*Sigma + Mu1  # Mu1的正态分布
        else:
            X[0, i] = np.random.normal()*Sigma + Mu2  # Mu2的正态分布
    if isdebug:
        print("***********")
        print(u"初始观測数据X：")
        print(X)

# EM算法：步骤1，计算E[zij]
def e_step(Sigma,k,N):
    global Expectations  # 记录整体期望值
    global Mu  # 记录整体模型的Mu
    global X  # 记录正态分布的点
    for i in range(0, N):  # 遍历这整体的N个点
        Denom = 0  # 记录期望值的分母
        # 计算期望值, 公式：http://img.blog.csdn.net/20140820211401271?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY2hhc2RtZW5n/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast
        # 与书本公式对应只是符号不同
        for j in range(0,k):
            # 计算分母
            Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
        for j in range(0,k):  # 遍历模型
            # 计算分子
            Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[0,i]-Mu[j]))**2)
            Expectations[i,j] = Numer / Denom  # 计算每个模型的对于每个点的期望值
    if isdebug:
        print("***********")
        print("隐藏变量E（Z）：")
        print(Expectations)
# EM算法：步骤2，求最大化E[zij]的參数Mu
def m_step(k,N):
    global Expectations  # 记录整体期望值
    global X  # 记录正态分布的点
    # 迭代Mu
    for j in range(0,k): # 遍历模型
        Numer = 0
        Denom = 0
        for i in range(0,N):
            Numer += Expectations[i,j]*X[0,i]  # 母子
            Denom += Expectations[i,j]  # 分母
        Mu[j] = Numer / Denom  # 迭代Mu

# 算法迭代iter_num次。或达到精度Epsilon停止迭代
def run(Sigma,Mu1,Mu2,k,N,iter_num,Epsilon):
    '''

    :param Sigma: 方差
    :param Mu1: 模型一的参数
    :param Mu2: 模型二的参数
    :param k: 混合模型包含k个高斯參数预计
    :param N: 每个模型有N个点
    :param iter_num: 最大迭代次数
    :param Epsilon: 控制退出的精确度阀值
    :return: Nan
    '''
    ini_data(Sigma,Mu1,Mu2,k,N)
    print(u"初始<u1,u2>:", Mu)
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)  # 记录迭代前的Mu
        e_step(Sigma,k,N)  # E步
        m_step(k,N)  # M步
        print(i,Mu)
        if sum(abs(Mu-Old_Mu)) < Epsilon:  # 迭代变化小于阀值, 退出
            break

if __name__ == '__main__':
   run(6, 40, 20, 2, 1000, 1000, 0.0001)
   plt.hist(X[0,:],50)
   plt.show()