import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as Data
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")


def linear_regress(features):
    # 线性拟合
    import matplotlib.pyplot as plt1
    import seaborn as sns
    import scipy.stats as stats
    import statsmodels.api as sm
    from scipy.stats import chi2_contingency
    fig = sns.lmplot(x='top',y='cth',data=features,
                   legend_out=False,#将图例呈现在图框内
                   truncate=False,#根据实际的数据范围，对拟合线做截断操作
                   # aspect=1.25, # 长宽比
                   size=10,
                   scatter_kws={"s": 5}
                   )
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # fig = plt1.figure(dpi=300, figsize=(10, 8))
    # ax = fig.add_subplot(111)
    #
    # Axis_line = np.linspace(*ax.get_xlim(), 2)
    # ax.plot(Axis_line, Axis_line, transform=ax.transAxes, color='gray', linestyle='--')

    plt1.xlim(0,None)
    plt1.ylim(0,None)
    # ax.scatter(x=features['top'], y=features['cth'],s=3 )
    # ax.set_ylim(bottom=0.)
    # ax.set_xlim(left=0.)
    plt1.xlabel("Cloudsat 云顶高度")
    plt1.ylabel("MODIS 云顶高度")



    cos = Cosine_Sim(features['top'], features['cth'])
    textstr = "CosineSimilarity="+str(format(cos,'.4f'))
    ymin, ymax = plt1.ylim()
    xmin, xmax = plt1.xlim()
    print(xmax,ymax)
    plt1.plot([0,max(xmax,ymax)],[0,max(xmax,ymax)],color='r',linewidth=2, linestyle='--')
    plt1.text(9000,4500, textstr, weight="bold", color="r",fontsize=16)
    # plt1.show()
    plt1.savefig("../../img/两种数据余弦相关性.png")

    fit=sm.formula.ols('cth~top',data=features).fit()
    print(fit.params) # Intercept：1985.440393  top：0.684590
    # cth = 1985.440393 + 0.684590 * top
    cth = 1985.440393 + 0.684590 * features['top']
    return cth

def model1(features,cth):
    # 画图比较

    # 设置布局
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10),sharey=True)
    fig.autofmt_xdate(rotation = 45)
    pointsize = 4

    # 标签值
    ax1.scatter(features['index'], features['top'],s=pointsize)
    ax1.set_xlabel('');ax1.set_ylabel('Cloud Top Height');ax1.set_title('CloudSat')

    #
    ax2.scatter(features['index'], features['cth'],s=pointsize)
    ax2.set_xlabel('');ax2.set_ylabel('Cloud Top Height');ax2.set_title('MODIS')

    #
    ax3.scatter(features['index'], features['top'],s=pointsize)
    ax3.set_xlabel('');ax3.set_ylabel('Cloud Top Height');ax3.set_title('CloudSat')

    # 修正
    ax4.scatter(features['index'], cth,s=pointsize)
    ax4.set_xlabel('');ax4.set_ylabel('Cloud Top Height');ax4.set_title('MODIS Linear regress')

    plt.tight_layout(pad=2)
    # plt.show()
    plt.savefig("../../img/两种数据云顶高度比较.png")

def Cosine_Sim(A,B):
    # 计算向量余弦值
    from numpy.linalg import norm
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    # print("Cosine Similarity:", cosine)
    return cosine

# 创建网络模型
from torch.nn import functional as F
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 64,bias=True),
            nn.Linear(64, 128,bias=True),
            nn.Linear(128, 256, bias=True),
            nn.Linear(256, 512, bias=True),
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 64, bias=True),
            nn.Linear(64, 32, bias=True),
            nn.Linear(32, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def model2(features,chunk_num,lossA):
    # 对云顶高度进行预报
    model_path = "model_082601.pkl"

    # 数据预处理
    features['index'] = range(len(features))
    # print(features.head())
    # features = features[(features['date'] == 20080101)] # 时间筛选
    # print(len(features))
    # print(features.shape)
    index1 = features[(features.top < 0)].index.tolist()
    features = features.drop(index=index1)
    index2 = features[(features.cth < 0)].index.tolist()
    features = features.drop(index=index2)

    # cloudsat 云顶高度
    labels = np.array(features['top'])
    # print(labels)
    # print("labels.dtype: ",labels.dtype)

    # 在特征值去掉标签
    features = features.drop('top', axis=1)
    features = features.drop('CloudLayer', axis=1)
    features = features.drop('type', axis=1)
    features = features.drop('date', axis=1)

    trains = features.loc[:, ['bt0', 'bt1', 'bt2', 'bt3', 'bt4', 'bt5', 'bt6']]
    trains = np.array(trains)
    # print('trains:')
    # print(trains)
    # print('trains_features',features.shape)
    # print("trains.dtype: ",trains.dtype)

    # return

    x = torch.from_numpy(trains.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    # print("x.dtype: ", x.dtype)
    # print("y.dtype: ", y.dtype)
    # print("x.shape: ", x.shape)

    # 批处理样本大小
    batch_size = 128

    # 将训练集转化为张量后，整理到一起
    train_data = Data.TensorDataset(x,y)
    # 定义一个数据加载器，将训练数据进行批量处理
    train_loader = Data.DataLoader(
        dataset=train_data, # 使用的数据集
        batch_size=batch_size, # 批处理样本大小
        shuffle=True, # 每次迭代前打乱数据
        num_workers=1, # 使用两个进程
    )
    # test_dataloader = Data.DataLoader(test_data, batch_size=batch_size)

    # length 长度
    train_data_size = len(train_data)
    # test_data_size = len(test_data)

    lr = 0.000002  # 学习率
    gamma = 0.9  # 动量

    # 实例化网络
    net = Model()  # 先初始化一个模型，这边的 Model() 指代你的 pytorch 模型
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))  # 再加载模型参数

    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        net = net.cuda()

    # 定义损失函数
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss(reduction = "mean")
    criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=gamma)

    loss_array = []
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        y = y.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        yhat = net(x)
        yhat = yhat.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        
        y.resize_(yhat.shape)
        # 查看预测值和真实值
        # for i in range(len(yhat)):
        #     print(yhat[i],y[i])
        loss = criterion(yhat, y)
        # print(batch_idx, "loss: ", float(loss),yhat[0],y[0])
        loss_array.append(float(loss))
        loss.backward()
        opt.step()
        opt.zero_grad()
    chunk_loss = sum(loss_array)/len(loss_array)
    lossA.append(chunk_loss)
    print("chunk", chunk_num, " loss: ", chunk_loss)
    torch.save(net.state_dict(),model_path)
    import csv
    with open('lossA.csv', 'a',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([str(chunk_loss)])
    return lossA



def get_data(header,nrows):
    # 读取数据
    names = np.array(pd.read_csv('../../data/DoneData/2008.csv', header=None, nrows=1))[0]
    features = pd.read_csv('../../data/DoneData/2008.csv', header=header, nrows=nrows, names=names)
    features['index'] = range(len(features))
    # print(features.head())
    # features = features[(features['date'] == 20080101)] # 时间筛选
    # print(len(features))
    # print(features.shape)
    index1 = features[(features.top < 0)].index.tolist()
    features = features.drop(index=index1)
    index2 = features[(features.cth < 0)].index.tolist()
    features = features.drop(index=index2)
    return features


if __name__ == '__main__':
    print("读取数据")
    # features = get_data(0,10000000)
    test_data = get_data(10001,5000)
    print('test_data.type: ',type(test_data))
    names = np.array(pd.read_csv('../../data/DoneData/2008.csv', header=None, nrows=1))[0]
    df = pd.read_csv('../../data/DoneData/2008.csv', header=0, chunksize=100000,names=names)
    print("数据读取完毕")
    print("准备训练")

    chunk_num = 0
    lossA = []
    modelx = Model()
    for chunk in df:
        # print('chunk.type: ',type(chunk))
        chunk_num += 1
        lossA = model2(chunk, chunk_num,lossA)
        # if chunk_num == 50:
        #     break

    import matplotlib.pyplot as plt
    plt.plot(lossA)
    plt.show()



    # print(features)
    # print(test_data)




    # 线性拟合
    # cth = linear_regress(features)
    # print("原数据余弦相似度：    ",Cosine_Sim(features['top'],features['cth']))
    # print("线性拟合后余弦相似度： ",Cosine_Sim(features['top'],cth))
    # 画图比较
    # model1(features,cth)
    # 神经网络预报云顶高度
    # model2(features,test_data)

    print("ok!")
