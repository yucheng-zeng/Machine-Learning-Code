import numpy as np
import scipy.stats


def em_single(observations,priors):

    """
    EM算法的单次迭代
    Arguments
    ------------
    priors:[theta_A,theta_B]
    observation:[m X n matrix]

    Returns
    ---------------
    new_priors:[new_theta_A,new_theta_B]
    :param priors:
    :param observations:
    :return:
    """
    # 用1表示H（正面），0表示T（反面）：
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    pi = priors[0]  # 对应π
    theta_A = priors[1]  # 对应p
    theta_B = priors[2]  # 对应q
    pi_list = []
    # E step
    for observation in observations:  # 遍历所有样本
        len_observation = len(observation)  # 记录样本的长度, 既是第i个样本抛的次数
        num_heads = observation.sum()  # 正面出现的次数
        num_tails = len_observation-num_heads  # 反面出现的次数

        # 二项分布求解公式
        contribution_A = scipy.stats.binom.pmf(num_heads,len_observation,theta_A)  # 求A硬币的二项分布
        contribution_B = scipy.stats.binom.pmf(num_heads,len_observation,theta_B)  # 求B硬币的二项分布
        #print(contribution_A)
        #print(contribution_B)
        # 将两个概率正规化，得到数据来自硬币A，B的概率：
        weight_A = contribution_A / (contribution_A + contribution_B)  # 求u
        weight_B = contribution_B / (contribution_A + contribution_B)
        #更新在当前参数下A，B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_heads  #
        counts['A']['T'] += weight_A * num_tails  #
        counts['B']['H'] += weight_B * num_heads  #
        counts['B']['T'] += weight_B * num_tails  #
        pi_list.append(weight_A)  # 记录u

    # M step
    pi = sum(pi_list)/len(pi_list)  # 更新pi值
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])  # 更新p值
    # print('A=%s'%new_theta_A)
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])  # 更新q值
    # print('B=%s'%new_theta_B)
    return [pi, new_theta_A, new_theta_B]

def em(observations,prior,tol = 1e-6,iterations=10000):
    """
    EM算法
    ：param observations :观测数据
    ：param prior：模型初值
    ：param tol：迭代结束阈值
    ：param iterations：最大迭代次数
    ：return：局部最优的模型参数
    """
    iteration = 0  # 记录迭代次数
    while iteration < iterations:
        new_prior = em_single(observations, prior)  # 记录迭代之后的值
        delta_change = np.abs(sum(prior)-sum(new_prior))  # 计算差值
        if delta_change < tol:
            break
        else:
            prior = new_prior  # 更新初值
            iteration +=1  #　迭代次数加一
    return [new_prior, iteration]  # 返回当前初值， 以及更新次数

if __name__=='__main__':

    # 观测到的数据, 硬币投掷结果
    observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                             [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                             [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                             [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                             [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
    theta = [0.5, 0.5, 0.6]  # 模型初值, 分别对应π,p,q
    print(em(observations, theta))