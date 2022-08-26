from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
import pprint
from HDFread import HDFread # 读取hdf文件
from decoder import decoderScenario

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


def readmac(src,date):
    # print(src)
    # src = '../data/MAC/20080101/MAC06S0.A2008001.0000.002.2017074151447.hdf'
    hdf = SD(src)
    lat = hdf.select('Latitude').get().flatten() # 纬度
    lon = hdf.select('Longitude').get().flatten() # 经度
    time = np.zeros(len(lat))
    time[:] = date # 日期
    bt0 = np.array(hdf.select('Brightness_Temperature').get()[0].flatten()) # 亮温
    bt1 = np.array(hdf.select('Brightness_Temperature').get()[1].flatten())
    bt2 = np.array(hdf.select('Brightness_Temperature').get()[2].flatten())
    bt3 = np.array(hdf.select('Brightness_Temperature').get()[3].flatten())
    bt4 = np.array(hdf.select('Brightness_Temperature').get()[4].flatten())
    bt5 = np.array(hdf.select('Brightness_Temperature').get()[5].flatten())
    bt6 = np.array(hdf.select('Brightness_Temperature').get()[6].flatten())
    ctt = np.array(hdf.select('Cloud_Top_Temperature').get().flatten()) # 云顶温度
    cth = np.array(hdf.select('Cloud_Top_Height').get().flatten()) # 云顶高度
    ctp = np.array(hdf.select('Cloud_Top_Pressure').get().flatten())  # 云顶压力
    th = np.array(hdf.select('Tropopause_Height').get().flatten())  # 对流层高度
    cf = np.array(hdf.select('Cloud_Fraction').get().flatten())  # 云量
    sft = np.array(hdf.select('Surface_Temperature').get().flatten()) # 表面温度
    sfp = np.array(hdf.select('Surface_Pressure').get().flatten()) # 表面压力

    df1 = pd.DataFrame({'lat': lat, 'lon': lon,'date':time,
                        'bt0': bt0, 'bt1': bt1,'bt2': bt2,'bt3': bt3,
                        'bt4': bt4, 'bt5': bt5, 'bt6': bt6,
                        'ctt': ctt,'cth': cth,'ctp': ctp,
                        'th': th,'cf': cf,
                        'sft': sft,'sfp': sfp})
    return df1

def readcloudsat1(csFilePath,cs_file_list):
    csclassdata = np.empty(shape=[0, 127], dtype=float)
    for path in cs_file_list:
        # file1 = "../../data/CloudSat/2008182053651_11561_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
        file1 = str(csFilePath + '\\' + path)
        hdf = SD(file1)
        type = decoderScenario(file1)
        lon = HDFread(file1, 'Longitude').flatten()
        lat = HDFread(file1, 'Latitude').flatten()
        csclass = np.arange(len(lon) * 127, dtype="float").reshape(len(lon), 127)
        csclass[:, 0] = lon
        csclass[:, 1] = lat
        for i in range(125):
            csclass[:, 2 + i] = type[:, i]
        csclassdata = np.append(csclassdata, csclass, axis=0)
    datacsDict = {}  # 把 cloudsat数据转换成字典
    for item in range(127):
        if item == 0:
            a = {'lon': csclassdata[:, 0]}
        elif item == 1:
            a = {'lat': csclassdata[:, 1]}
        else:
            a = {'type' + str(item - 2): csclassdata[:, item]}
        datacsDict.update(a)
    # print(datacsDict)
    datacs = pd.DataFrame(datacsDict)
    return datacs

def readcloudsat(csFilePath,cs_file_list):
    csclassdata = np.empty(shape=[0, 5], dtype=float)
    for path in cs_file_list:
        # file1 = "../../data/CloudSat/2008182053651_11561_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
        file1 = str(csFilePath + '\\' + path)
        hdf = SD(file1)
        # type = decoderScenario(file1)
        lon = HDFread(file1, 'Longitude').flatten()
        lat = HDFread(file1, 'Latitude').flatten()
        CloudLayerType = hdf.select('CloudLayerType').get()  # 读取CloudLayerType
        CloudLayerTop = hdf.select('CloudLayerTop').get()  # 读取CloudLayerTop
        # 读取CloudLayer
        hdfobj = HDF(file1, HC.READ)
        vs = hdfobj.vstart()
        v = hdfobj.vgstart()
        layertype_index = vs.find('CloudLayer')  # cloudlayer数据存放位置
        nrec = vs.attach(layertype_index).inquire()[0]  # 数据总数
        CloudLayer = np.array(vs.attach(layertype_index).read(nrec)).flatten()  # 取出所有数据
        csclass = np.arange(len(lon) * 5, dtype="float").reshape(len(lon), 5)

        csclass[:, 0] = lon
        csclass[:, 1] = lat
        csclass[:, 2] = CloudLayer
        csclass[:, 3] =CloudLayerType[:, 0]
        csclass[:, 4] = CloudLayerTop[:, 0]
        # for i in range(10):
        #     csclass[:, 3 + i] = CloudLayerType[:, i]
        # for i in range(10):
        #     csclass[:, 13 + i] = CloudLayerTop[:, i]
        csclassdata = np.append(csclassdata, csclass, axis=0)
    datacsDict = {}  # 把 cloudsat数据转换成字典
    for item in range(5):
        if item == 0:
            a = {'lon': csclassdata[:, 0]}
        elif item == 1:
            a = {'lat': csclassdata[:, 1]}
        elif item == 2:
            a = {'CloudLayer': csclassdata[:, 2]}
        elif item == 3:
            a = {'type': csclassdata[:, 3]}
        else:
            a = {'top': csclassdata[:, 4]}
        # elif item > 2 and item <= 12:
        #     a = {'type' + str(item - 3): csclassdata[:, item]}
        # else:
        #     a = {'top' + str(item - 13): csclassdata[:, item]}
        datacsDict.update(a)


    # print(datacsDict)
    datacs = pd.DataFrame(datacsDict)
    datacs['CloudLayer'] = datacs['CloudLayer'].astype(int)
    datacs['type'] = datacs['type'].astype(int)
    # for item in range(0,11):
    #     if item == 10:
    #         datacs['CloudLayer'] = datacs['CloudLayer'].astype(int)
    #     else:
    #         datacs['type' + str(item)] = datacs['type' + str(item)].astype(int)
    return datacs

def DAYlist(start,end):
    yearmonthday = pd.date_range(start,end,freq="D").strftime("%Y%m%d").to_list()
    return yearmonthday


if __name__ == '__main__':
    # pass
    days = DAYlist('2008-01-01','2008-12-31')
    for date in days:
        # date = "20080101"
        print('读取',date,'的数据')
        macFilePath = fr'E:\Code\python\cloudclassfication\data\MAC\{date}'
        mac_file_list = os.listdir(macFilePath)
        cloudsatFilePath = fr'E:\Code\python\cloudclassfication\data\CloudSat\{date}'
        cloudsat_file_list = os.listdir(cloudsatFilePath)
        # print(mac_file_list)

        # MAC数据
        df0 = pd.DataFrame({})
        for i in range(1, len(mac_file_list)):
            macPath = "../../data/MAC/" + date + '/' + mac_file_list[i]
            # print(macPath)
            df1 = readmac(macPath,date)
            df0 = pd.concat([df0, df1])
            # print(mac_file_list[i] + " ok")
        mac_path = '../../data/DoneData/MAC/MAC' + date +'.csv'
        df0.to_csv(mac_path,index=False)
        # print("共有数据",df0.shape[0],"条")

        dfcs = readcloudsat(cloudsatFilePath,cloudsat_file_list)
        cs_path = '../../data/DoneData/CloudSat/CS' +date +'.csv'
        dfcs.to_csv(cs_path,index=False)
        print("ok")
        # break
    print("全部读取完毕!")