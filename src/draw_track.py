import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap  # 导入Basemap库
from location import getLonAndLat


# 画扫描轨道图
def plotTrack():
    # 可视化
    #绘制地图 加入站点
    plt.figure(figsize=(8, 4))  #它的参数figsize=(16,8)定义了图的大小。
    m = Basemap()  #使用Basemap()创建一个地图
    m.drawcoastlines()  #把海岸线画上
    m.drawcountries(color='grey', linewidth=1)  # 开始画上国家

    # 填充陆地、胡泊、海洋的颜色
    m.fillcontinents(
        color='g',  # 陆地颜色
        lake_color='b',  # 湖泊颜色
        alpha=0.2)
    # m.drawmapboundary(fill_color='blue')    # 填充海洋

    # 添加经纬线
    m.drawmeridians(
        np.arange(0, 360, 30),  # 设置纬线的其实范围，以及维度的间隔
        color='pink',  # 颜色
        linewidth=0.5,  # 线宽
        labels=[1, True, 0, True],
        fontsize=10,
    )
    m.drawparallels(
        np.arange(-90, 90, 30),
        color='green',  # 颜色
        linewidth=0.5,  # 线宽
        labels=[1, True, 0, 1],
        fontsize=10,
    )

    # 轨道列表
    filePath = [
        '../data/UjNpYC_0001_0005/2007051220514_04348_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf'
    ]

    for fi in range(len(filePath)):
        longitude,latitude = getLonAndLat(filePath[fi])
        m.scatter(longitude,latitude)
    plt.show()

if __name__ == '__main__':
    plotTrack()