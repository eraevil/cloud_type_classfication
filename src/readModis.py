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
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

file = "../data/MODIS/MOD06_L2.A2008183.1455.061.2017292031620.hdf"
hdf = SD(file)
longitude = hdf.select('Longitude').get()
latitude = hdf.select('Latitude').get()
file2 = '../data/UjNpYC_0001_0005/2007051220514_04348_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf'
print('longitude',longitude)
print('latitude',latitude)
exit()

# 通过cartopy获取底图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))

# 用经纬度对地图区域进行截取，这里只展示我国沿海区域
ax.set_extent([85, 170, -20, 60], crs=ccrs.PlateCarree())
ax.set_extent([95, 125, -5, 25], crs=ccrs.PlateCarree())
ax.set_extent([120, 125, -2, 5], crs=ccrs.PlateCarree())

# 设置名称
ax.set_title('MODIS_06 & Cloudsat CLDCLASS 地理空间匹配', fontsize=16)

# 设置地图属性，比如加载河流、海洋
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAKES, alpha=0.5)

# 画经纬度网格
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1.2, color='k', alpha=0.3, linestyle='--')
gl.xlabels_top = False  # 关闭顶端的经纬度标签
gl.ylabels_right = False  # 关闭右侧的经纬度标签
gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
#设置经纬度网格的间隔
gl.xlocator = mticker.FixedLocator(np.arange(95, 125, 5))
gl.ylocator = mticker.FixedLocator(np.arange(-5, 25, 5))
# 设置显示范围
ax.set_extent([95, 125, -5, 25],crs=ccrs.PlateCarree())
# 设置坐标标签
ax.set_xticks(list(range(95, 130, 5)), crs=ccrs.PlateCarree())
ax.set_yticks(list(range(-5, 30, 5)), crs=ccrs.PlateCarree())
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# print(longitude)
print(longitude.flatten())
# print(latitude)
longitude = longitude.flatten()
latitude = latitude.flatten()
scatter = ax.scatter(longitude, latitude,
               s= 0.05,
               c='red', alpha=0.8,linewidths = 2,
               # c=data.depth / data.depth.max(), alpha=0.8,
               transform=ccrs.PlateCarree())


filePath = [
    # '../data/UjNpYC_0001_0005/2008182021905_11559_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf',
    # '../data/UjNpYC_0001_0005/2008182035758_11560_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf',
    '../data/UjNpYC_0001_0005/2008182053651_11561_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf',
    # '../data/UjNpYC_0001_0005/2008182071544_11562_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf',
    '../data/UjNpYC_0001_0005/2009231162830_17611_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf',
]
for fi in range(len(filePath)):
    longitude2 = HDFread(filePath[fi], 'Longitude')
    latitude2 = HDFread(filePath[fi], 'Latitude')
    longitude2 = np.array(longitude2)
    latitude2 = np.array(latitude2)
    scatter = ax.scatter(longitude2, latitude2,
                         s=0.001,
                         c='blue', alpha=0.8,linewidths = 0.4,
                         # c=data.depth / data.depth.max(), alpha=0.8,
                         transform=ccrs.PlateCarree())

plt.show()
# fig.savefig("../img/modis和cloudsat地理匹配.png",dpi = 2000)
