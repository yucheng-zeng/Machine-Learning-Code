import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(0)  # 保证随机的唯一性
# 创建数据集, 线性可分：
array = np.random.randn(20, 2)
X = np.r_[array-[3, 3],array+[3, 3]]  # 创建x的集合
y = [0]*20+[1]*20  # 设置标签
print(X)

# 建立svm模型
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
''' 
C:当C趋近于无穷大时：意味着分类严格不能有错误
  当C趋近于很小的时：意味着可以有更大的错误容忍
gamma:gamma值越大，映射的维度越高，模型越复杂
'''
svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)


x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
print('X[:, 0]=%s'%X[:, 0])
# 得到向量w  : w_0x_1+w_1x_2+b=0
w = clf.coef_[0]
f = w[0]*xx1 + w[1]*xx2 + clf.intercept_[0] + 1  # 加1后才可绘制 -1 的等高线 [-1,0,1] + 1 = [0,1,2]
plt.contour(xx1, xx2, f, [0, 1, 2], colors='r')  # 绘制分隔超平面、H1、H2
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], color='k')  # 绘制支持向量点
plt.show()
