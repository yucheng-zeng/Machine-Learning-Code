import os
import pickle as pkl
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import binary_logistic as bl


class bow:
    def __init__(self,dict={}):
        self.dict = dict

    def build_dict(self, path):
        txts = os.listdir(path)
        index = len(self.dict.keys())
        for txt in txts:
            root = path+'/'+txt
            for word in self.read_txt(root):
                if word in self.dict.keys():
                    continue
                else:
                    self.dict[word] = index
                    index += 1

    def to_vec(self, txt):
        vec = [0]*len(self.dict.keys())
        for word in txt:
            vec[self.dict[word]] += 1
        return vec

    def read_txt(self,path):
        punctuation = [',', ',', '.', "'", '"', ')', '(', '-', '?', '!']
        txt = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                for word in line:
                    word = word.lower()
                    for p in punctuation:
                        word = word.replace(p, '')
                    txt.append(word)
        return txt

    def save_model(self, path='./email/bow.pk'):
        with open(path, 'wb') as f:
            pkl.dump(self.dict, f)
            print('保存模型成功')

    def load_model(self, path='./email/bow.pk'):
        if os.path.exists(path):
            f = open(path, 'rb')
            self.dict = pkl.load(f)
            print('加载模型成功')
        else:
            print('加载模型失败')


if __name__ == '__main__':
    model = bow()
    model.build_dict('./email/ham')
    model.build_dict('./email/spam')
    model.save_model()

    # 加载数据
    ptxts = os.listdir('./email/ham')  # 正例
    pvec = []
    for p in ptxts:
        root = './email/ham/'+p
        vec = model.to_vec(model.read_txt(root))
        pvec.append(vec)
    plabel = np.array([1]*len(ptxts))

    ntxts = os.listdir('./email/spam')  # 负例
    nvec = []
    for n in ntxts:
        root = './email/spam/'+n
        vec = model.to_vec(model.read_txt(root))
        nvec.append(vec)
    nlabel = np.array([0]*len(ntxts))

    pvec = np.array(pvec)
    nvec = np.array(nvec)

    vec = np.vstack((pvec, nvec))
    label = np.hstack((plabel, nlabel))
    # print(vec.shape)

    train_features, test_features, train_labels, test_labels = train_test_split(vec, label, test_size=0.3, random_state=23323)

    # 训练
    lr = bl.LogisticRegression(max_iteration=1)

    print('Start training')
    lr.train(train_features, train_labels, test_features, test_labels)  # 训练

    # 预测
    print('Start predicting')
    test_predict = lr.predict_dataSet(test_features)  # 开始预测

    # 结果
    bl.measure(test_labels, test_predict)




