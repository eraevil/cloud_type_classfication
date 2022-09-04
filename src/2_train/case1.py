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
    # print(xmax,ymax)
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
            nn.Linear(8, 64,bias=True),nn.ReLU(),
            # nn.Linear(64, 128, bias=True), nn.ReLU(),
            # nn.Linear(128, 64, bias=True), nn.ReLU(),
            # nn.Linear(64, 32, bias=True), nn.ReLU(),
            nn.Linear(64, 128,bias=True),nn.ReLU(),
            nn.Linear(128, 256, bias=True),nn.ReLU(),
            nn.Linear(256, 512, bias=True),nn.ReLU(),
            nn.Linear(512, 1024, bias=True),nn.ReLU(),
            nn.Linear(1024, 512, bias=True),nn.ReLU(),
            nn.Linear(512, 256, bias=True),nn.ReLU(),
            nn.Linear(256, 128, bias=True),nn.ReLU(),
            nn.Linear(128, 64, bias=True),nn.ReLU(),
            nn.Linear(64, 32, bias=True),nn.ReLU(),
            nn.Linear(32, 1, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 去除非法数据
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
    index15 = chunk[(chunk.top < 0)].index.tolist() # top -99000
    index = index1 + index2 + index3 + index4 + index5 + index6 + index7 + index8 + index9 + index10 + index11 + index12 + index13 + index14 + index15

    chunk = chunk.drop(index=index)

    # print(len(chunk))
    return chunk

def model2(features,chunk_num,lossA,lr):
    # 对云顶高度进行预报
    model_path = "./model/model_090406.pkl"

    # 数据预处理
    features['index'] = range(len(features))

    # trains = features.loc[:, ['bt0', 'bt1', 'bt2', 'bt3', 'bt4', 'bt5', 'bt6']]
    trains = features.loc[:,['bt0', 'bt1', 'bt2', 'bt3', 'bt4', 'bt5', 'bt6', 'ctt', 'cth', 'ctp', 'cf', 'th', 'sft', 'sfp', 'top']]
    trains = sifter_data(trains)  # 祛除脏数据

    # cloudsat 云顶高度
    labels = np.array(trains['top'])
    trains = trains.drop('top', axis=1) # 在训练集中去掉云顶高度

    trains = pca_data(trains) # pca主成分分析
    trains = np.array(trains)
    # print('train shape ',trains.shape)

    # 数据归一化
    from sklearn import preprocessing
    trains = preprocessing.MinMaxScaler().fit_transform(trains)
    # return

    x = torch.from_numpy(trains.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))

    # 批处理样本大小
    batch_size = 32

    # 将训练集转化为张量后，整理到一起
    train_data = Data.TensorDataset(x,y)
    # 定义一个数据加载器，将训练数据进行批量处理
    train_loader = Data.DataLoader(
        dataset=train_data, # 使用的数据集
        batch_size=batch_size, # 批处理样本大小
        shuffle=True, # 每次迭代前打乱数据
        num_workers=8, # 使用两个进程
    )
    # test_dataloader = Data.DataLoader(test_data, batch_size=batch_size)

    # length 长度
    train_data_size = len(train_data)
    # test_data_size = len(test_data)

    # lr = 0.0000001  # 学习率
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
    # criterion = nn.SmoothL1Loss()
    opt = optim.SGD(net.parameters(), lr=lr, momentum=gamma)

    # 截断梯度
    # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25)

    loss_array = []
    for batch_idx, (x, y) in enumerate(train_loader):
        opt.zero_grad()
        x = x.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        y = y.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        yhat = net(x)
        yhat = yhat.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))

        # yhat[yhat == float("Inf")] = 0
        y.resize_(yhat.shape)
        # 查看预测值和真实值
        # for i in range(len(yhat)):
        #     print(yhat[i],y[i])
        loss = criterion(yhat, y)
        # print(batch_idx, "loss: ", float(loss),yhat[0],y[0])



        loss_array.append(float(loss))
        loss.backward()

        # 检查权重和梯度是否在更新
        # for params in net.named_parameters():
        #     [name, param] = params
        #
        #     if param.grad is not None:
        #         print(name, end='\t')
        #         print('weight:{}'.format(param.data.mean()), end='\t')
        #         print('grad:{}'.format(param.grad.data.mean()))

        opt.step()

    chunk_loss = sum(loss_array)/len(loss_array)
    lossA.append(chunk_loss)
    print("chunk", chunk_num, " lr: ",lr ," loss: ", chunk_loss)
    torch.save(net.state_dict(),model_path)
    import csv
    with open('./loss/loss_090406.csv', 'a',newline='') as f:
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
    # index1 = features[(features.top < 0)].index.tolist()
    # features = features.drop(index=index1)
    # index2 = features[(features.cth < 0)].index.tolist()
    # features = features.drop(index=index2)
    return features

def check_loss_curve():
    df = pd.read_csv('./loss/loss_090406.csv')
    # print(df.info)
    import matplotlib.pyplot as plt
    # plt.scatter(range(len(df)),df)
    plt.plot(df)
    titlestr = "第" + str(i) + "次训练后损失值曲线"
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(titlestr)
    plt.show()

def train(epoch):
    df = pd.read_csv('../../data/DoneData/2008.csv', header=0, chunksize=1000000, names=names)
    print("第",epoch,"次训练 数据读取完毕")
    print("准备训练")

    # lr = 1.25e-09
    #lr = 0.00000001  # 基础学习率
    lr = 0.00001
    # lr = lr / (2 * (epoch+1))

    chunk_num = 0
    lossA = []
    modelx = Model()
    for chunk in df:
        # print('chunk.type: ',type(chunk))
        chunk_num += 1
        # if chunk_num % 30 == 0:
        #     lr = lr / 2

        # TODO LIST
        lossA = model2(chunk, chunk_num, lossA,lr) # 训练10万条数据
        # break
        # if chunk_num == 50:
        #     break
    check_loss_curve() # 查看累计损失曲线

# PCA主成分分析
def pca_data(chunk):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    data = chunk.loc[:, ['bt0', 'bt1', 'bt2', 'bt3', 'bt4', 'bt5', 'bt6', 'ctt', 'cth', 'ctp', 'cf', 'th', 'sft', 'sfp']]
    # 数据标准化
    scaler = StandardScaler()
    scaler.fit(data)
    X = scaler.transform(data)

    # 主成分pca拟合
    model = PCA()
    model.fit(X)

    # 主成分核载矩阵
    # model.components_
    columns = ['PC' + str(i) for i in range(1, 15)]
    pca_loadings = pd.DataFrame(model.components_, columns=data.columns, index=columns)
    # print(round(pca_loadings, 2))

    # 计算每个样本的主成分得分
    pca_scores = model.transform(X)
    pca_scores = pd.DataFrame(pca_scores, columns=columns)
    # print(pca_scores.shape)
    # print(pca_scores.head())

    model = PCA(n_components=8)
    model.fit(X)
    X_train_pca = model.transform(X)
    
    return X_train_pca
    # return X_train_pca

if __name__ == '__main__':
    print("读取数据")
    # features = get_data(0,10000000)
    # test_data = get_data(10001,5000)
    # print('test_data.type: ',type(test_data))
    names = np.array(pd.read_csv('../../data/DoneData/2008.csv', header=None, nrows=1))[0]
    from time import time

    t = time()  # 开始时间
    print("开始计时")

    for i in range(10):
        train(i)
        # break

    print('时间消耗：%.2f秒' % (time() - t))

    # import matplotlib.pyplot as plt
    # plt.plot(lossA)
    # plt.show()



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
