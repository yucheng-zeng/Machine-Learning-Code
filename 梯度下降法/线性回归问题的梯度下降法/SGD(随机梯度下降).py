import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


'''
随机梯度下降算法每次只随机选择一个样本来更新模型参数，因此每次的学习是非常快速的，并且可以进行在线更新
'''
# 构造训练数据
#x = np.arange(-10, 10., 0.5)  # 根据start与stop指定的范围以及step设定的步长, 获去数据集, 返回一个列表
x = np.array([0.25, 1.00, 2.25, 4.00, 6.25])
m = len(x)  # 训练数据点数目
#print(m)
x0 = np.full(m, 1.0)  # 获取一个长度为m, 每个元素都是1.0 的列表
# T 表示转秩矩阵, 不加T 第一行元素为依次为x0,第二行元素依次为x, 得到2*m矩阵
input_data = np.vstack([x0, x]).T  # 构造矩阵, 第一列元素为依次为x0,第二列元素依次为x,得到m*2矩阵
#target_data = x * 2 + 5 + np.random.randn(m)  # 随机设置随即设置x对应的y值
target_data = np.array([1.338, 5.091, 11.304, 20.003, 31.201])  # 随机设置随即设置x对应的y值
print(target_data)

# 两种终止条件
loop_max = 100  # 最大迭代次数(防止死循环)
epsilon = 1e-5  # 目标函数与拟合函数的距离当的距离小于epsilo时, 退出

# 初始化权值
np.random.seed(0)  #
theta = np.random.randn(input_data.shape[1])  # 初始化theta,, 生成一个列表,维度与输入空间一致

alpha = 0.1  # 步长(注意取值过大会导致振荡即不收敛,过小收敛速度变慢)
diff = 0.0  # 记录梯度
oldtheta = np.zeros(2)  # 记录上一次迭代的theta
count = 0  # 循环次数
finish = 0  # 终止标志

while count < loop_max:
    count += 1

    # 随机梯度下降的权值是通过考查某个训练样例来更新的

    # 遍历训练数据集，不断更新权值
    for i in range(m):
        # 对应公式：https://img-blog.csdn.net/20170617205722973?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHVhaHVhemh1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center
        # 以下数求损失函数的梯度
        diff = (np.dot(theta, input_data[i]) - target_data[i]) * input_data[i]
        print(np.dot(theta, input_data[i].T))
        print((np.dot(theta, input_data[i])))
        print('h=%s' % diff)
        print(target_data[i])
        theta = theta - alpha * diff  # 注意步长alpha的取值,过大会导致振荡,过小收敛速度太慢

    # 判断是否已收敛
    if np.linalg.norm(theta - oldtheta) < epsilon:  # 求范数, 如果两个theta差距过小, 表示误差足够小
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
plt.plot(x, x * theta[1] + theta[0], 'r')
plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.ylabel('转动惯量(10^-3 kg·m^2)')
plt.xlabel('距离的平方(10^2 mm^2)')
plt.show()
