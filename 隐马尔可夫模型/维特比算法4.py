import numpy as np

def viterbi(trainsition_probability,emission_probability,pi,obs_seq):
    #转换为矩阵进行运算
    trainsition_probability=np.array(trainsition_probability)
    emission_probability=np.array(emission_probability)
    pi=np.array(pi)
    obs_seq = [0, 2, 3]
    # 最后返回一个Row*Col的矩阵结果
    Row = np.array(trainsition_probability).shape[0]
    Col = len(obs_seq)
    #定义要返回的矩阵
    F=np.zeros((Row,Col))
    #初始状态
    F[:,0]=pi*np.transpose(emission_probability[:,obs_seq[0]])
    for t in range(1,Col):
        list_max=[]
        for n in range(Row):
            list_x=list(np.array(F[:,t-1])*np.transpose(trainsition_probability[:,n]))
            #获取最大概率
            list_p=[]
            for i in list_x:
                list_p.append(i*10000)
            list_max.append(max(list_p)/10000)
        F[:,t]=np.array(list_max)*np.transpose(emission_probability[:,obs_seq[t]])
    return F

if __name__=='__main__':
    #隐藏状态
    invisible=['Sunny','Cloud','Rainy']
    #初始状态
    pi=[0.63,0.17,0.20]
    #转移矩阵
    trainsion_probility=[[0.5,0.375,0.125],[0.25,0.125,0.625],[0.25,0.375,0.375]]
    #发射矩阵
    emission_probility=[[0.6,0.2,0.15,0.05],[0.25,0.25,0.25,0.25],[0.05,0.10,0.35,0.5]]
    #最后显示状态
    obs_seq=[0,2,3]
    #最后返回一个Row*Col的矩阵结果
    F=viterbi(trainsion_probility,emission_probility,pi,obs_seq)
    print(F)