import matplotlib.image as image
import numpy as np
import matplotlib.pyplot as plt


def paint(img1,img2,img3,img4,FR2, FR3, FR4, SN2, SN3, SN4):
    ax1 = plt.subplot(221)
    ax1.imshow(img1)
    ax1.set_title('Original Picture')
    ax2 = plt.subplot(222)
    ax2.imshow(img2)
    ax2.set_title('K=1,FR='+str(FR2)+',PSNR='+str(SN2))
    ax3 = plt.subplot(223)
    ax3.imshow(img3)
    ax3.set_title('K=3,FR='+str(FR3)+',PSNR='+str(SN3))
    ax4 = plt.subplot(224)
    ax4.imshow(img4)
    ax4.set_title('K=5,FR='+str(FR4)+',PSNR='+str(SN4))
    plt.show()

def Feature_Ratio(eigVals, eigValInd):
    after = 0
    origin = float(np.sum(eigVals))
    for i in eigValInd:
        after += eigVals[i]
    after = float(after)
    return after/origin

def PSNR(originImg, newImg):
    Red_MSE = 0
    Green_MSE = 0
    Blue_MSE = 0
    for i in range(0, originImg.shape[0]):
        Red_MSE += (originImg[i, 0] - newImg[i, 0]) ** 2
        Green_MSE += (originImg[i, 1] - newImg[i, 1]) ** 2
        Blue_MSE += (originImg[i, 2] - newImg[i, 2]) ** 2
    MSE = (Red_MSE + Green_MSE + Blue_MSE)/3.0
    Result = 10*np.log10(255**2/MSE)
    return Result
# 将数据降维
def pca(dataMat, topNfeat=9999999):
    '''
    :param dataMat: 矩阵型原始数据
    :param topNfeat: 保留的特征个数
    :return:
    '''
    SN = 0
    meanVals = np.mean(dataMat, axis=0)  # 所有行对应维度相加, 然后除以行数, 的到每一个维度的平均值
    meanRemoved = dataMat - meanVals  # 原数据集移除均值
    covMat = np.dot(meanRemoved, meanRemoved.T)  # 计算协方差矩阵
    # 计算特征值eigVals, 计算特征向量eigVects
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)  # 排序, 从小到达排序
    # 后面的-1代表的是将值倒序，原来特征值从小到大，现在从大到小
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 获取指定个特征值最大的下标
    # redEigVects = eigVects[eigValInd, :]  # 获取前eigValInd个特征值最大的特征向量
    redEigVects = eigVects[:, eigValInd]  # 获取前eigValInd个特征值最大的特征向量
    lowDDataMat = redEigVects.T * meanRemoved # 将数据转换到新空间中, 降维之后的数据集
    reconMat = (redEigVects * lowDDataMat) + meanVals  # 降维后的数据再次映射到原来空间中，用于与原始数据进行比较
    return lowDDataMat, reconMat, round(Feature_Ratio(eigVals, eigValInd), 2), round(PSNR(dataMat, reconMat),2).real

def encoding(img):
    r = []
    g = []
    b = []
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
                r.append(img[i, j, 0])
                g.append(img[i, j, 1])
                b.append(img[i, j, 2])
    return np.mat(r).T, np.mat(g).T, np.mat(b).T, img.shape[0], img.shape[1]

def decoding(r, g, b, w, h):
    img = np.zeros([w, h, 3])
    for i in range(0, w):
        for j in range(0, h):
            img[i, j, 0] = int(r[i*w+j, 0])
            img[i, j, 1] = int(g[i*w+j, 0])
            img[i, j, 2] = int(b[i*w+j, 0])
    img = img.astype('uint8')
    return img

def isSame(mat1, mat2, w, h, d):
    for i in range(0, w):
        for j in range(0, h):
            for k in range(0, d):
                if mat1[i, j, k] != mat2[i, j, k]:
                    return False
    return True


def test(dataMat, k=1):
    lowDDataMat, reconMat, FR, SN = pca(dataMat, k)
    r = reconMat[:, 0]
    g = reconMat[:, 1]
    b = reconMat[:, 2]
    newImg = decoding(r, g, b, w, h)
    return newImg, FR, SN

if __name__ == '__main__':
    path = './cat.jpg'
    img = image.imread(path)
    r, g, b, w, h = encoding(img)
    dataMat = np.hstack((np.hstack((r, g)), b))
    newImg2, FR2, SN2 = test(dataMat, 1)
    newImg3, FR3, SN3 = test(dataMat, 3)
    newImg4, FR4, SN4 = test(dataMat, 5)
    print(SN2, SN3, SN4)
    paint(img, newImg2, newImg3, newImg4, FR2, FR3, FR4, SN2, SN3, SN4)


