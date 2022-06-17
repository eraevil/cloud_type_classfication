# cartopy：用来获取地图
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# matplotlib：用来绘制图表
import matplotlib.pyplot as plt
# shapely：用来处理点线数据
import shapely.geometry as sgeom
import warnings
import re
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def x1():
    x = np.arange(-5, 5, 0.1)
    y = x * 3
    fig = plt.figure() # 1个窗口简化写法
    # fig = plt.figure(num=1, figsize=(15, 8),dpi=80)     #详细写法，设置大小，分辨率
    ax1 = fig.add_subplot(2,1,1)  # 同fig.add_subplot(211)通过fig添加子图，参数：行数，列数，第几个。1个可省略
    # 括号里面的值前两个是轴域原点坐标（从左下角计算的），后两个是显示坐标轴的长度。
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) #add_axes()生成子图的灵活性更强，完全可以实现add_subplot()方法的功能
    # ax2 = fig.add_subplot(2,1,2)  #通过fig添加子图，参数：行数，列数，第几个。1个可省略
    # ax1 = fig.add_subplot(111, facecolor='r') #设置背景颜色
    plt.plot(x, y) # 画图
    plt.scatter(x, y, c='r') # 设置颜色
    plt.xlabel('X Axis') #设置x轴名称 ax1.set_xlabel
    plt.ylabel('Y Axis') #设置Y轴名称 ax1.set_ylabel
    plt.title('this is a demo') # 设置标题
    plt.xlim((-5, 5)) # 设置横轴范围 ax1.set_xlim 可略，默认x,y 所有值范围
    plt.ylim((-15, 15)) # 设置纵轴范围 ax1.set_ylim 可略
    plt.savefig('d:/f.png') # 保存图片，plt.savefig('aa.jpg', dpi=400, bbox_inches='tight') dpi分辨率，bbox_inches子图周边白色空间的大小
    # 左右下上4个轴
    # 设置轴的位置
    ax1.spines['left'].set_position('center')
    # 设置轴的颜色
    ax1.spines['right'].set_color('none')
    # 设置轴的位置
    ax1.spines['bottom'].set_position('center')
    # 设置轴的颜色
    ax1.spines['top'].set_color('none')
    # 显示网格。which参数的值为major(只绘制大刻度)、minor(只绘制小刻度)、both，默认值为major。axis为'x','y','both'
    ax1.grid(b=True, which='major', axis='both', alpha=0.5, color='skyblue', linestyle='--', linewidth=2)
    # ax1.set_xticks([])  # 去除坐标轴刻度
    ax1.set_xticks((-5, -3, -1, 1, 3, 5))  # 设置坐标轴刻度
    ax1.text(2.8, 7, r'y=3*x')  # 指定位置显示文字,plt.text()
    axes1 = plt.axes([.2, .3, .1, .1], facecolor='y')  # 在当前窗口添加一个子图，rect=[左, 下, 宽, 高]，是使用的绝对布局，不和以存在窗口挤占空间
    axes1.plot(x, y)  # 在图上画子图
    ax1.annotate('important point', xy=(2, 6), xytext=(3, 1.5),  # 添加标注，参数：注释文本、指向点、文字位置、箭头属性
                              arrowprops=dict(facecolor='black', shrink=0.05),)
    plt.show() #所有窗口运行

    ax1.invert_yaxis() # 反转y轴

def draw_coastal():
    # 通过cartopy获取底图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 用经纬度对地图区域进行截取，这里只展示我国沿海区域
    # ax.set_extent([85, 170, -20, 60], crs=ccrs.PlateCarree())

    # 设置名称
    ax.set_title('2017年台风路径图', fontsize=16)

    # 设置地图属性，比如加载河流、海洋
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAKES, alpha=0.5)

    typhoonData = pd.read_csv('../data/typhoonData.csv')
    # 先对台风编号进行循环，提取单个台风数据
    for typhoonNumber in typhoonData['台风编号'].unique():
        typhoon = typhoonData[typhoonData['台风编号'] == typhoonNumber]
        # 再对单个台风数据进行处理，提取经纬度
        for typhoonPoint in np.arange(len(typhoon) - 1):
            lat_1 = typhoon.iloc[typhoonPoint, 3] / 10
            lon_1 = typhoon.iloc[typhoonPoint, 4] / 10
            lat_2 = typhoon.iloc[typhoonPoint + 1, 3] / 10
            lon_2 = typhoon.iloc[typhoonPoint + 1, 4] / 10
            point_1 = lon_1, lat_1
            point_2 = lon_2, lat_2
            # 最后可视化
            ax.add_geometries([sgeom.LineString([point_1, point_2])], crs=ccrs.PlateCarree(), edgecolor='red')

    # 展示地图
    plt.show()

if __name__ == '__main__':
    draw_coastal()