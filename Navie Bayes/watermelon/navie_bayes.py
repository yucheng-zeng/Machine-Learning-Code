import numpy as np


def load_data(filepath):
    '''
    :arg filepath  filepath是数据的路径
    :fun 加载数据：1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
    :return 加载后的数据
    '''

    train_data = []  # 训练数据
    file_object = open(filepath, encoding='UTF-8')
    file_object.readline()
    while 1:
        data = file_object.readline()
        if not data:
            break
        else:
            train_data.append(data)
    file_object.close()
    test = []
    for s in train_data:
        test.append(s.replace('\n', '').split(','))  # 去掉\n和把数据按照’,‘分割再存
    return test


def count_labels(data):
    '''

    :param data:数据集
    :return: 返回好瓜和坏瓜的数目
    '''
    yes = 0
    no = 0
    for s in range(data.__len__()):
        if data[s][-1] == '是':
            yes += 1
        else:
            no += 1
    return yes, no


def handle_one_data(data, attr, location, yes, no, attr_dis):
    '''
    :param data: 数据集
    :param attr: 要传入的属性
    :param location: 传入属性的位置
    :param yes: 好瓜数量
    :param no: 坏瓜数量
    :param attr_dis: 各个属性的取值不同的个数
    :return: 返回该属性在好瓜或者是坏瓜的前提下的概率
    '''
    attr_y, attr_n = 0, 0
    for s in range(data.__len__()):
        if data[s][-1] == '是':
            if data[s][location] == attr:
                attr_y += 1
        else:
            if data[s][location] == attr:
                attr_n += 1
    return (attr_y + 1) / (yes + attr_dis[location-1]), (attr_n + 1) / (no + attr_dis[location-1])


def handle_data(data):
    '''

    :param data: 数据集
    :return: 对密度和含糖率的均值和标准差
    '''
    midu_y = []  # 好瓜密度
    tiandu_y = []  # 好瓜甜度
    midu_n = []  # 非好瓜密度
    tiandu_n = []  # 非好瓜甜度
    for s in range(data.__len__()):
        if data[s][-1] == '是':
            midu_y.append(np.float(data[s][-3]))
            tiandu_y.append(np.float(data[s][-2]))
        else:
            midu_n.append(np.float(data[s][-3]))
            tiandu_n.append(np.float(data[s][-2]))
    m_midu_y = np.mean(midu_y)
    m_midu_n = np.mean(midu_n)
    t_tiandu_y = np.mean(tiandu_y)
    t_tiandu_n = np.mean(tiandu_n)
    std_midu_y = np.std(midu_y)
    std_midu_n = np.std(midu_n)
    std_tiandu_y = np.std(tiandu_y)
    std_tiandu_n = np.std(tiandu_n)

    return m_midu_y, m_midu_n, t_tiandu_y, t_tiandu_n, std_midu_y, std_midu_n, std_tiandu_y, std_tiandu_n


def show_result(p_yes, p_no):
    '''

    :param p_yes: 在好瓜的前提下，测试数据各个属性的概率
    :param p_no: 在是坏瓜的前提下，测试数据的各个属性的概率
    :return: 是好瓜或者是坏瓜
    '''
    p1 = 1.0
    p2 = 1.0
    for s in range(p_yes.__len__()):
        p1 *= np.float(p_yes[s])
        p2 *= np.float(p_no[s])
    if p1 > p2:
        print("好瓜", p1, p2)
    else:
        print("坏瓜", p1, p2)


def count_attr_dis(data):
    '''
    :param data: 数据集
    :return: 各个属性取值的个数
    '''
    count = []  # 记录各个属性的取值有多少个不同
    for i in range(data[0].__len__()):
        if i == 0 or i == 7 or i == 8:  # 去掉编号，密度，甜度这个属性
            continue
        d = []
        for s in range(data.__len__()):
            if not d.__contains__(data[s][i]):  # 如果读到的属性不包含在d里就加入到d中
                d.append(data[s][i])
        count.append(d.__len__())  # 统计属性取值不同的个数
    return count


if __name__ == '__main__':
    filepath = 'bayes.txt'  # 数据文件路径
    data = load_data(filepath)  # 加载数据
    attr_dis = count_attr_dis(data)  # 求出不同属性的取值

    m_midu_y, m_midu_n, t_tiandu_y, t_tiandu_n, std_midu_y, std_midu_n, std_tiandu_y, std_tiandu_n = handle_data(data)
    yes, no = count_labels(data)
    p_yes = [(yes+ 1)/(yes + no + attr_dis[6])]
    p_no = [(no+ 1)/(yes + no + attr_dis[6])]
    test_data = ['青绿', '蜷缩', '清脆', '清晰', '凹陷', '硬滑', 0.697, 0.460]

    for s in range(6):  # 求出各个属性的概率
        s_yes, s_no = handle_one_data(data, test_data[s], s + 1, yes, no, attr_dis)
        p_yes.append(s_yes)
        p_no.append(s_no)

    # 求西瓜书公式（7.18）
    p_yes.append(
        1 / (np.sqrt(2 * np.pi) * std_midu_y) * np.exp((-1) * ((test_data[6] - m_midu_y) ** 2) / std_midu_y ** 2))
    p_no.append(
        1 / (np.sqrt(2 * np.pi) * std_midu_n) * np.exp((-1) * ((test_data[6] - m_midu_n) ** 2) / std_midu_n ** 2))

    p_yes.append(
        1 / (np.sqrt(2 * np.pi) * std_tiandu_y) * np.exp((-1) * ((test_data[7] - t_tiandu_y) ** 2) / std_tiandu_y ** 2))
    p_no.append(
        1 / (np.sqrt(2 * np.pi) * std_tiandu_n) * np.exp((-1) * ((test_data[7] - t_tiandu_n) ** 2) / std_tiandu_n ** 2))

    print(p_yes)
    print(p_no)
    show_result(p_yes, p_no)

    # 防止某个属性的取值个数为0的概率出现，采用拉皮拉斯修正(各个属性不同取值已经完成如函数count_attr_dis)

    print(count_attr_dis(data))
