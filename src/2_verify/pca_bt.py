import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch import nn
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from mpl_toolkits import mplot3d

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def sifter_data(chunk):
    # print(len(chunk))
    # 'bt0', 'bt3', 'bt4', 'bt5', 'bt6', 'ctt', 'cth', 'ctp', 'cf', 'th', 'sft', 'sfp'
    index1 = chunk[(chunk.bt0 < 0) | (chunk.bt0 > 20000)].index.tolist() #bt0 0-20000  -32768
    index13 = chunk[(chunk.bt1 < 0) | (chunk.bt1 > 20000)].index.tolist() # bt1 0-20000  -32768
    index14 = chunk[(chunk.bt2 < 0) | (chunk.bt2 > 20000)].index.tolist() # bt2 0-20000  -32768
    index2 = chunk[(chunk.bt3 < 0) | (chunk.bt3 > 20000)].index.tolist() #bt3 0-20000  -32768
    index3 = chunk[(chunk.bt4 < 0) | (chunk.bt4 > 20000)].index.tolist() #bt4 0-20000  -32768
    index4 = chunk[(chunk.bt5 < 0) | (chunk.bt5 > 20000)].index.tolist() #bt5 0-20000  -32768
    index5 = chunk[(chunk.bt6 < 0) | (chunk.bt6 > 20000)].index.tolist() #bt6 0-20000  -32768
    index6 = chunk[(chunk.ctt < 0) | (chunk.ctt > 20000)].index.tolist() #ctt 0-20000  -32768
    index7 = chunk[(chunk.cth < 0) | (chunk.cth > 18000)].index.tolist() #cth 0-18000  -32767
    index8 = chunk[(chunk.ctp < 10) | (chunk.ctp > 11000)].index.tolist() #ctp 10-11000  -32768
    index9 = chunk[(chunk.cf < 0) | (chunk.cf > 100)].index.tolist() #cf 0-100  127
    index10 = chunk[(chunk.th < 10) | (chunk.th > 11000)].index.tolist() #th 10-11000  -32768
    index11 = chunk[(chunk.sft < 0) | (chunk.sft > 20000)].index.tolist() #sft 0-20000  -32768
    index12 = chunk[(chunk.sfp < 8000) | (chunk.sfp > 11000)].index.tolist() #sfp 8000-11000  -32768
    index = index1 + index2 + index3 + index4 + index5 + index6 + index7 + index8 + index9 + index10 + index11 + index12 + index13 + index14

    chunk = chunk.drop(index=index)

    # print(len(chunk))
    return chunk

if __name__ == '__main__':
    print("读取数据")
    # features = get_data(0,10000000)
    # test_data = get_data(10001,5000)
    # print('test_data.type: ',type(test_data))
    names = np.array(pd.read_csv('../../data/DoneData/2008.csv', header=None, nrows=1))[0]
    from time import time

    t = time()  # 开始时间
    print("开始计时")


    ##### 参数
    n_component = 7
    k = 3
    # oop方式


    # df = pd.read_csv('../../data/DoneData/2008.csv', header=0, chunksize=50000000, names=names)
    df = pd.read_csv('../../data/DoneData/2008.csv', header=0, chunksize=50000000, names=names)
    for chunk in df:
        # print('chunk.type: ',type(chunk))
        num = 14
        data = chunk.loc[:, ['bt0', 'bt1', 'bt2', 'bt3', 'bt4', 'bt5', 'bt6', 'ctt', 'cth', 'ctp', 'cf', 'th', 'sft', 'sfp']]
        data = sifter_data(data) # 祛除脏数据

        pd.options.display.max_columns = 10
        # 变量之间相关性计算
        corrV = round(data.corr(), 2)
        print(corrV)
        sns.heatmap(corrV, annot=True) # 热力图

        plt.show()

        # 数据标准化
        scaler = StandardScaler()
        scaler.fit(data)
        X = scaler.transform(data)

        fig = plt.figure()
        # type(fig)
        fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(8, 10))

        # 主成分pca拟合
        model = PCA()
        model.fit(X)
        # 每个主成分能解释的方差
        #model.explained_variance_
        # 每个主成分能解释的方差的百分比
        #model.explained_variance_ratio_
        # 可视化
        ax2.plot(model.explained_variance_ratio_, 'o-')
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Proportion of Variance Explained')
        ax2.set_title('PVE')


        #  画累计百分比，这样可以判断选几个主成分
        ax3.plot(model.explained_variance_ratio_.cumsum(), 'o-')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Cumulative Proportion of Variance Explained')
        ax3.axhline(0.9, color='k', linestyle='--', linewidth=1)
        ax3.set_title('Cumulative PVE')

        # 主成分核载矩阵
        model.components_
        columns = ['PC' + str(i) for i in range(1, num+1)]
        pca_loadings = pd.DataFrame(model.components_, columns=data.columns, index=columns)
        print(round(pca_loadings, 2))

        plt.show()
        # plt.savefig('ok.png')

        #该矩阵展示了每个主成分是原始数据的线性组合，以及线性的系数
        #画图展示
        fig, ax = plt.subplots(7, 2)
        plt.subplots_adjust(hspace=1, wspace=0.5)
        for i in range(1, 15):
            ax = plt.subplot(7, 2, i)
            ax.plot(pca_loadings.T['PC' + str(i)], 'o-')
            ax.axhline(0, color='k', linestyle='--', linewidth=1)
            # ax.set_xticks(range(7))
            ax.set_xticklabels(data.columns, rotation=30)
            ax.set_title('PCA Loadings for PC' + str(i))

        plt.show()

        #计算每个样本的主成分得分
        pca_scores = model.transform(X)
        pca_scores = pd.DataFrame(pca_scores, columns=columns)
        print(pca_scores.shape)
        print(pca_scores.head())

        # sns.scatterplot(x='PC1', y='PC2', data=pca_scores)
        # plt.title('Biplot')

        plt.show()

        # 得到PCA主成分
        X_train_pca = model.transform(X)
        print(X_train_pca)


        break
