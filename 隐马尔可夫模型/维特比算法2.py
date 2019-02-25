import numpy as np
# -*- codeing:utf-8 -*-
__author__ = 'youfei'

#   隐状态
hidden_state = ['sunny', 'rainy']

#   观测序列
obsevition = ['walk', 'shop', 'clean']


#   根据观测序列、发射概率、状态转移矩阵、发射概率
#   返回最佳路径
def compute(obs, states, start_p, trans_p, emit_p):

    #   max_p（3*2）每一列存储第一列不同隐状态的最大概率
    max_p = np.zeros((len(obs), len(states)))

    #   path（2*3）每一行存储上max_p对应列的路径
    path = np.zeros((len(states), len(obs)))

    #   初始化
    for i in range(len(states)):  # 遍历隐状态
        max_p[0][i] = start_p[i] * emit_p[i][obs[0]]
        path[i][0] = i

    for t in range(1, len(obs)):  # 遍历观测序列
        newpath = np.zeros((len(states), len(obs)))  # 新路径
        for y in range(len(states)):  # 遍历每个隐因子
            prob = -1
            for y0 in range(len(states)):
                nprob = max_p[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]
                if nprob > prob:
                    prob = nprob
                    state = y0
                    #   记录路径
                    max_p[t][y] = prob
                    for m in range(t):
                        newpath[y][m] = path[state][m]
                    newpath[y][t] = y

        path = newpath

    max_prob = -1
    path_state = 0
    #   返回最大概率的路径
    for y in range(len(states)):
        if max_p[len(obs)-1][y] > max_prob:
            max_prob = max_p[len(obs)-1][y]
            path_state = y

    return path[path_state]


state_s = [0, 1]  # 隐状态
obser = [0, 1, 2]  # 观测序列

#   初始状态，测试集中，0.6概率观测序列以sunny开始
start_probability = [0.6, 0.4]

#   转移概率，0.7：sunny下一天sunny的概率
transititon_probability = np.array([[0.7, 0.3], [0.4, 0.6]])

#   发射概率，0.4：sunny在0.4概率下为shop
emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

result = compute(obser, state_s, start_probability, transititon_probability, emission_probability)

for k in range(len(result)):
    print(hidden_state[int(result[k])])