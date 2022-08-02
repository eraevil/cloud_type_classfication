import pandas as pd
import transbigdata as tbd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os
import numpy as np
import warnings

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(3, 1, 1)
ax1 = fig.add_subplot(3, 1, 2)
ax2 = fig.add_subplot(3, 1, 3)
ax.set_title('2008年1月1日 MAC与Cloudsat的地理匹配', fontsize=16)
NUM = 50

# 按文件名读取整个文件
# data = pd.read_csv("../../data/DoneData/20080101temp.csv").head(50)
cs_data = pd.read_csv("../../data/DoneData/CloudSat/CS20080101.csv") # 原始cloudsat数据
mac_data = pd.read_csv("../../data/DoneData/MAC/MAC20080101.csv") # 原始Mac数据
ver_data = pd.read_csv("../../data/DoneData/20080101temp.csv") # 按Mac对cloudsat进行标记后数据
matched_data = pd.read_csv("../../data/DoneData/20080101.csv") # 匹配后数据
cs_data = cs_data[(cs_data['lat']>=-0.46) & (cs_data['lat']<=0) & (cs_data['lon']>=20.64) & (cs_data['lon']<=20.76)]
mac_data = mac_data[(mac_data['lat']>=-0.46) & (mac_data['lat']<=0) & (mac_data['lon']>=20.64) & (mac_data['lon']<=20.76)]
ver_data = ver_data[(ver_data['lat_x']>=-0.46) & (ver_data['lat_x']<=0) & (ver_data['lon_x']>=20.64) & (ver_data['lon_x']<=20.76)]
matched_data = matched_data[(matched_data['lat_x']>=-0.46) & (matched_data['lat_x']<=0) & (matched_data['lon_x']>=20.64) & (matched_data['lon_x']<=20.76)]




if __name__ == '__main__':
    lat = mac_data['lat']
    lon = mac_data['lon']
    lat1 = cs_data['lat']
    lon1 = cs_data['lon']
    lat2 = ver_data['lat_x']
    lon2 = ver_data['lon_x']
    lat3 = ver_data['lat_y']
    lon3 = ver_data['lon_y']
    lat4 = matched_data['lat_x']
    lon4 = matched_data['lon_x']
    lat5 = matched_data['lat_y']
    lon5 = matched_data['lon_y']
    # 原始
    scatter = ax.scatter(lon, lat,
                         s=2,
                         c='blue', alpha=1,linewidths = 1,)
    scatter = ax.scatter(lon1, lat1,
                         s=2,
                         c='red', alpha=1, linewidths=1,)
    # 按modis像元分类
    scatter = ax1.scatter(lon, lat,
                         s=2,
                         c='blue', alpha=1, linewidths=1, )
    scatter = ax1.scatter(lon2, lat2,
                         s=2,
                         c='red', alpha=1, linewidths=1, )
    scatter = ax1.scatter(lon3, lat3,
                         s=500,
                         c='violet', alpha=0.3, linewidths=1, )
    # 完成匹配
    scatter = ax2.scatter(lon4, lat4,
                          s=2,
                          c='red', alpha=1, linewidths=1, )
    scatter = ax2.scatter(lon5, lat5,
                         s=2,
                         c='blue', alpha=1, linewidths=1, )
    scatter = ax2.scatter(lon5, lat5,
                          s=500,
                          c='violet', alpha=0.3, linewidths=1, )

    ver_tempdata = np.array(ver_data)
    for i in range(len(ver_tempdata)):
        x_list = []
        y_list = []
        x_list.append(ver_tempdata[i][0])
        x_list.append(ver_tempdata[i][129])
        y_list.append(ver_tempdata[i][1])
        y_list.append(ver_tempdata[i][128])
        ax1.plot(x_list, y_list, color='black', linewidth=1, alpha=0.6)

    matched_tempdata = np.array(matched_data)
    for i in range(len(matched_tempdata)):
        x_list = []
        y_list = []
        x_list.append(matched_tempdata[i][0])
        x_list.append(matched_tempdata[i][129])
        y_list.append(matched_tempdata[i][1])
        y_list.append(matched_tempdata[i][128])
        ax2.plot(x_list, y_list, color='black', linewidth=1, alpha=0.6)

    # ax.set_extent([20, 21,-1, 1])
    # plt.show()
    fig.savefig("../../img/按MODIS像元对CloudSat进行匹配的过程", dpi=2000)
    print("ok")

    # print("开始。。。。")
    # # cs_data = cs_data[(cs_data['lat'] >= -2) & (cs_data['lat'] <= 19) & (cs_data['lon'] >= 99) & (cs_data['lon'] <= 125)]
    #
    # lat1 = cs_data['lat']
    # lon1 = cs_data['lon']
    # print('lon', lon1.shape)
    # print('lat', lat1)
    # scatter = ax.scatter(lon1, lat1,
    #                      s=2,
    #                      c='red', alpha=1, linewidths=1, transform=ccrs.PlateCarree())
    # ax.set_title('Cloudsat数据', fontsize=16)
    # # plt.show()
    # fig.savefig("../../img/Cloudsat数据.png", dpi=2000)
    # # cs_data.to_csv('../../data/DoneData/20080101nanhai.csv', index=False)
    # print("ok!!")