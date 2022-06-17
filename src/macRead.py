from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
import pprint
from HDFread import HDFread # 读取hdf文件

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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
ax.set_title('2008年1月1日 MAC与Cloudsat的地理匹配', fontsize=16)
# 设置地图属性，比如加载河流、海洋
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1.2, color='k', alpha=0.3, linestyle='--')
gl.xlabels_top = False  # 关闭顶端的经纬度标签
gl.ylabels_right = False  # 关闭右侧的经纬度标签
gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
#设置经纬度网格的间隔
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
gl.ylocator = mticker.FixedLocator(np.arange(-90, 90, 30))
ax.set_xticks(list(range(-180, 180, 30)), crs=ccrs.PlateCarree())
ax.set_yticks(list(range(-90, 90, 30)), crs=ccrs.PlateCarree())


def main():
    date = "20080101"
    macFilePath = fr'E:\Code\python\cloudclassfication\data\MAC\{date}'
    mac_file_list = os.listdir(macFilePath)
    cloudsatFilePath = fr'E:\Code\python\cloudclassfication\data\CloudSat\{date}'
    cloudsat_file_list = os.listdir(cloudsatFilePath)
    # print(mac_file_list)

    for i in range(2,len(mac_file_list)):
        macPath = "../data/MAC/" + date + '/'+ mac_file_list[i]
        readmac(macPath)
        print(mac_file_list[i] + " ok")
        # break
    print("MODIS all ok!")
    for i in range(0, len(cloudsat_file_list)):
        cloudsatPath = cloudsatFilePath + '\\' + cloudsat_file_list[i]
        readcloudsat(cloudsatPath)
        print(cloudsat_file_list[i] + " ok")
        # break
        pass

    print("Cloudsat all ok!")
    # plt.show()
    # fig.savefig("../img/2008年1月1日 MAC与Cloudsat的地理匹配.png", dpi=2000)

def readmac(src):
    # print(src)
    # src = '../data/MAC/20080101/MAC06S1.A2008001.1400.002.2017074155411.hdf'
    hdf = SD(src)
    lat = hdf.select('Latitude').get().flatten()
    lon = hdf.select('Longitude').get().flatten()
    scatter = ax.scatter(lon, lat,
                         s=0.001,
                         c='red', alpha=1,linewidths = 1,
                         # c=data.depth / data.depth.max(), alpha=0.8,
                         transform=ccrs.PlateCarree())
def readcloudsat(src):
    longitude2 = HDFread(src, 'Longitude')
    latitude2 = HDFread(src, 'Latitude')
    longitude2 = np.array(longitude2)
    latitude2 = np.array(latitude2)
    scatter = ax.scatter(longitude2, latitude2,
                         s=0.001,
                         c='blue', alpha=1, linewidths=0.5,
                         # c=data.depth / data.depth.max(), alpha=0.8,
                         transform=ccrs.PlateCarree())
    pass


if __name__ == '__main__':
    print("start...")
    # main()

    # src = '../data/MAC/20080101/MAC06S1.A2008001.2355.002.2017074162514.hdf'
    src = '../data/MAC/20080101x/MAC06S1.A2008001.2355.002.2017074162514.hdf'
    src = '../data/MAC/20080101/MAC06S0.A2008001.2355.002.2017074162514.hdf'
    hdf = SD(src)
    data = hdf.datasets()
    for i in data:
        print(i)  # 具体类别

    # 变量查看
    lat = np.array(hdf.select('Latitude').get()) # 纬度
    lon = np.array(hdf.select('Longitude').get()) # 经度
    # bt = np.array(hdf.select('Brightness_Temperature').get()) # 亮温
    # ctt = np.array(hdf.select('Cloud_Top_Temperature').get()) # 云顶温度
    # cwp = np.array(hdf.select('Cloud_Water_Path').get()) # 云水路径
    # print(bt.shape)
    # print(ctt.shape)
    # print(lon.shape)
    # print(cwp.shape)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.scatter(lon.flatten(), lat.flatten(), s=3, c='blue', alpha=1, linewidths=0.5, )

    print('lon',lon[0])
    print('lat',lat)
    # long = lon.flatten()
    # lati = lat.flatten()
    # print(long)
    # print(lati)

    ax3 = fig.add_subplot(3, 1, 3)
    for i in range(25):
        long = lon[i]
        lati = lat[i]
        ax3.scatter(long,lati,s=3,c='blue', alpha=1, linewidths=0.5,)

    plt.show()
    # fig.savefig("../img/2008年1月1日 MAC与Cloudsat的地理匹配.png", dpi=2000)

    print("ok!")