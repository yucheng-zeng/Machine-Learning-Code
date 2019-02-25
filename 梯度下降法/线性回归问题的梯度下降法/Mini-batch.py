import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
'''
Mini-batch梯度下降综合了batch梯度下降与stochastic梯度下降，
在每次更新速度与更新次数中间取得一个平衡，其每次更新从训练集中随机选择b,b<m个样本进行学习
'''

# 构造训练数据
x = np.arange(-10, 10., 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
m = len(x)  # 训练数据点数目
#print(m)
x0 = np.full(m, 1.0)  # 获取一个长度为m, 每个元素都是1.0 的列表
# T 表示转秩矩阵, 不加T 第一行元素为依次为x0,第二行元素依次为x, 得到2*m矩阵
input_data = np.vstack([x0, x]).T  # 构造矩阵, 第一列元素为依次为x0,第二列元素依次为x,得到m*2矩阵
target_data = x * 2 + 5+ np.random.randn(m)  # 随机设置随即设置x对应的y值


# 两种终止条件
loop_max = 10000  # 最大迭代次数(防止死循环)
epsilon = 1e-3  # 目标函数与拟合函数的距离当的距离小于epsilo时, 退出

# 初始化权值
np.random.seed(0)  #
theta = np.random.randn(input_data.shape[1])  # 初始化theta,, 生成一个列表,维度与输入空间一致

alpha = 0.001  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
diff = 0.  # 记录梯度
oldtheta = np.zeros(2)  # 记录上一次迭代的theta
count = 0  # 循环次数
finish = 0  # 终止标志
minibatch_size = 5  # 每次更新的样本数

while count < loop_max:
    count += 1

    # minibatch梯度下降是在权值更新前对所有样例汇总误差，而随机梯度下降的权值是通过考查某个训练样例来更新的
    # 在minibatch梯度下降中，权值更新的每一步对多个样例求和，需要更多的计算

    randomIndexs = list(random.sample(range(0,m-1),k=minibatch_size))  # 从样本之中随机选择minibatch_size个点
    sum_m = np.zeros(2)
    for i in randomIndexs:
        # 对应公式：https://img-blog.csdn.net/20170617210032599?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHVhaHVhemh1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center            # 以下数求损失函数的梯度
        # 以下数求损失函数的梯度
        diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
        # 可以在迭代theta的时候乘以步长alpha, 也可以在梯度求和的时候乘以步长alpha
        sum_m = sum_m + diff  # 当alpha取值过大时,sum_m会在迭代过程中会溢出

    #if np.linalg.norm(sum_m)<epsilon:  # 设置阀值, 如果梯度过小, 退出
    #    break

    theta = theta - alpha * sum_m  # 注意步长alpha的取值,过大会导致振荡,过小收敛速度太慢

    # 判断是否已收敛
    if np.linalg.norm(theta - oldtheta) < epsilon:  # 求范数, 如果两个theta差距过小,表示误差足够小
        finish = 1  # 终止
        break
    else:
        oldtheta = theta  # 记录上一次迭代的theta
    print('loop count = %d' % count, '\tw:', theta)
print('loop count = %d' % count, '\tw:', theta)

# check with scipy linear regression
# intercept 回归曲线的截距
# slope 回归曲线的斜率
slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x, target_data)
print('intercept = %s slope = %s' % (intercept, slope))

plt.plot(x, target_data, 'g.')
plt.plot(x, theta[1] * x + theta[0], 'r')
plt.show()