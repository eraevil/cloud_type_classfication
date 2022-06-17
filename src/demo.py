import pprint

from numpy import byte
from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import cm, colors
# import seaborn as sns
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter,LatitudeLocator

from decoder import scenariolist2type # 导入云类型解码器
from location import segmentLat

def HDFread(filename, variable, Class=None):
    """
    Extract the data for non-scientific data in V mode of hdf file
    """
    hdf = HDF(filename, HC.READ)

    # Initialize the SD, V and VS interfaces on the file.
    sd = SD(filename)
    vs = hdf.vstart()
    v  = hdf.vgstart()

    # Found the class id
    if Class == None:
        ref = v.find('Geolocation Fields') # The default value for Geolocation fields
    else:
        ref = v.find(Class)

    # Open all data of the class
    vg = v.attach(ref)
    # All fields in the class
    members = vg.tagrefs()

    nrecs = []
    names = []
    for tag, ref in members:
        # Vdata tag
        try:
            vd = vs.attach(ref)
        except:
            continue
        # nrecs, intmode, fields, size, name = vd.inquire()
        nrecs.append(vd.inquire()[0])  # number of records of the Vdata
        names.append(vd.inquire()[-1]) # name of the Vdata
        vd.detach()

    idx = names.index(variable)
    var = vs.attach(members[idx][1])
    V   = var.read(nrecs[idx])
    var.detach()
    # Terminate V, VS and SD interfaces.
    v.end()
    vs.end()
    sd.end()
    # Close HDF file.
    hdf.close()

    return np.array(V)

def redCloudData(file):
    # file = "../data/UjNpYC_0001_0005/2008182071544_11562_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
    # hdf = SD(file)
    # cloudtype = hdf.select('CloudLayerType')
    longitude = HDFread(file,'Longitude')
    latitude = HDFread(file,'Latitude')
    longitude = np.array(longitude)
    # pprint.pprint(longitude)
    # pprint.pprint(latitude)
    # print('cloud type',cloudtype.get())
    sst = list(HDFread(file,'SST','Data Fields'))
    return longitude,latitude,sst

# 解码云类型
def decoderScenario(filepath):
    hdf = SD(filepath)
    cloudscenario = hdf.select('cloud_scenario')
    cloudscenarioV = cloudscenario.get()
    cloudscenarioV_set = set(cloudscenarioV.flatten()) # 拉伸数组，提取云类型编码值表
    typeDict = scenariolist2type(cloudscenarioV_set) # 得到云类型编码字典
    # 通过字典对云类型进行解码
    for i in typeDict:
        cloudscenarioV[cloudscenarioV == i] = typeDict[i]
    # print(cloudscenarioV)
    return cloudscenarioV



# 画云类型散点图
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
# 画云类型散点图
def draw_scatter(x, y,xl,yl, title, c, marker_size=500, savefig_name=""):
    # 创建画图窗口
    fig = plt.figure(1, figsize=(10, 4))
    # 将画图窗口分成2x2, 选择第一块区域作子图
    # subplot1 = fig.add_subplot(2, 2, 1)
    # 画散点图
    plt.scatter(x, y, s=marker_size, c=c, marker='o',linewidths=0.5)
    # 画参考线
    # subplot1.plot((0, 300), (0, 300), linestyle="--", linewidth=0.8, color="b")
    # 调整坐标轴范围
    # plt.xlim((0, len(x)))
    # plt.ylim((0, len(y)))
    plt.xlim((0, xl))
    plt.ylim((0, yl))
    # plt.ylim((0, 10))
    # 设置坐标轴刻度
    # xticks = np.arange(0, 126, 50)
    # yticks = np.arange(0, 37083, 50)
    # plt.xticks(xticks)
    # plt.yticks(yticks)
    # 设置标题
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Level')
    # 倒转y轴
    ax = plt.gca()
    ax.invert_yaxis()
    # 添加网格线
    # plt.grid(linestyle='--', color='grey')
    # 全局修改字体
    plt.rc('font', family='Times New Roman')
    # 显示色带
    # cbar_ticks = ['None','Ci','As','Ac','St','Sc','Cu','Ns','Dc']
    # cbar_ticks = c
    font = {'family': 'Times New Roman',
            'weight': 'bold',
            'size': 12}

    # cbar = plt.colorbar(orientation='horizontal', extend="both", pad=0.2)  # 显示色带
    # cbar.set_label("Cloud Type", fontdict=font)

    # cbar.set_ticks(cbar_ticks)
    # cbar.ax.tick_params(which="major", direction="in", length=2, labelsize=6)  # 主刻度
    # cbar.ax.tick_params(which="minor", direction="in", length=0)  # 副刻度
    # save figure
    # fig.tight_layout()
    # if "" != savefig_name.strip():
    #     plt.savefig(savefig_name, dpi=600)

    # plt.colorbar(label='CloudType')
    plt.show()

# plotCloudData()
filePath = '../data/UjNpYC_0001_0005/2007051220514_04348_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf'
type = decoderScenario(filePath)
vec = segmentLat(filePath,0,45) # 范围截取：纬度-60，60
type = type[vec]

# hdf = SD(filePath)
# type = hdf.select('CloudLayerType').get()
print(type,len(type),len(type[0]))
# exit(0)
# typecolor = type*20

# 文档配色
colors = dict([(0,'#ffffff'),(1,'#6e00c3'),(2,'#0025f0'),(3,'#007cd5'),(4,'#00bb63'),(5,'#9aff00'),(6,'#ffff00'),(7,'#ff7800'),(8,'#d91700')])
# 官网配色
# colors = dict([(0,'#000000'),(1,'#5400a3'),(2,'#0000ff'),(3,'#00a7ff'),(4,'#00e831'),(5,'#ffff00'),(6,'#ffaa00'),(7,'#f73c0a'),(8,'#c800c8')])
# colors = ['000000','5400a3','0000ff','00a7ff','00e831','ffff00','ffaa00','f73c0a','c800c8']
typecolor = type.flatten()
typecolor = np.array(typecolor, dtype='S7')
numarr = np.array(range(0,9), dtype='S7')
for i in colors:
    print(i,colors[i],numarr[i])
    typecolor[typecolor == numarr[i]] = colors[i]

# typecolor = np.log10(typecolor)
typecolor = np.array(typecolor, dtype='str')
# print(typecolor)
# exit()
x = range(len(type)) # 37082
y = range(len(type[1])) # 125
xv,yv = np.meshgrid(x,y,indexing="ij")
xx = np.array(xv.flat)
yy = np.array(yv.flat)
title = "Image of Cloudtype " + filePath[25:44]
draw_scatter(xx,yy,len(x),len(y),title,c=typecolor)

# plt.show()
