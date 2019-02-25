from numpy import *
import numpy as np
import random
import copy


# HMM-viterbi算法
def hmm_viterbi():
    Nstate = 3
    Nobs = 2
    T = 4
    init_prob = [1.0, 0.0, 0.0]
    trans_prob = np.array([[0.4, 0.6, 0.0], [0.0, 0.8, 0.2], [0.0, 0.0, 1.0]])
    emit_prob = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
    obs_seq = [0, 1, 0, 1]  # (ABAB)
    # 单独计算t=1时刻的局部概率
    partial_prob = zeros((Nstate, T))
    path = zeros((Nstate, T))
    for i in range(Nstate):
        partial_prob[i, 0] = init_prob[i] * emit_prob[i, obs_seq[0]]
        path[i, 0] = i

    # 计算t>1时刻的局部概率
    for t in range(1, T, 1):
        newpath = zeros((Nstate, T))
        for i in range(Nstate):
            prob = -1.0
            for j in range(Nstate):
                nprob = partial_prob[j, t - 1] * trans_prob[j, i] * emit_prob[i, obs_seq[t]]
                if nprob > prob:
                    prob = nprob
                    partial_prob[i, t] = nprob
                    newpath[i, 0:t] = path[j, 0:t]
                    newpath[i, t] = i
        path = newpath
    prob = -1.0
    j = 0
    print(path)
    for i in range(Nstate):
        if (partial_prob[i, T - 1] > prob):
            prob = partial_prob[i, T - 1]
            j = i
    print(path[j, :])


if __name__ == '__main__':
    hmm_viterbi()