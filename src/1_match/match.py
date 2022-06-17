# 导入基础库
from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
import pandas as pd
import os
import csv

# 导入自定义库
from HDFread import HDFread # 读取hdf文件
from decoder import decoderScenario

# 匹配粒度
granularity = 3 # 坐标保留小数位数
# 数据文件列表
date = "20080101"
modisFilePath = fr'E:\Code\python\cloudclassfication\data\MODIS\{date}'
modis_file_list = os.listdir(modisFilePath)
csFilePath = fr'E:\Code\python\cloudclassfication\data\CloudSat\{date}'
cs_file_list = os.listdir(csFilePath)

# MODIS06数据获取
modisdata = np.empty(shape=[0,9],dtype=float)
for path in modis_file_list:
    file = str(modisFilePath + '\\' + path)
    # file = "../../data/MODIS/MOD06_L2.A2008183.1455.061.2017292031620.hdf"
    print(file)
    hdf = SD(file)
    longitude = hdf.select('Longitude').get().flatten() # 经度
    latitude = hdf.select('Latitude').get().flatten() # 纬度
    modis = []
    modis = np.arange(len(longitude)*9,dtype="float").reshape(len(longitude),9)
    modis[:,0] = longitude
    modis[:,1] = latitude
    for i in range(7):
        bt = hdf.select('Brightness_Temperature').get()[i].flatten()  # 亮温数据
        modis[:,2+i] = bt
    # print(modis.shape)
    modisdata = np.append(modisdata,modis,axis=0)

# CloudSat CLDCLASS数据获取
csclassdata = np.empty(shape=[0,127],dtype=float)
for path in cs_file_list:
    # file1 = "../../data/CloudSat/2008182053651_11561_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
    file1 = str(csFilePath + '\\' + path)
    hdf = SD(file1)
    type = decoderScenario(file1)
    lon = HDFread(file1,'Longitude').flatten()
    lat = HDFread(file1,'Latitude').flatten()
    csclass = np.arange(len(lon)*127,dtype="float").reshape(len(lon),127)
    csclass[:,0] = lon
    csclass[:,1] = lat
    for i in range(125):
        csclass[:,2+i] = type[:,i]
    csclassdata = np.append(csclassdata, csclass, axis=0)

# 调整坐标粒度
modisdata[:,0] = np.round(modisdata[:,0],granularity)
modisdata[:,1] = np.round(modisdata[:,1],granularity)
csclassdata[:,0] = np.round(csclassdata[:,0],granularity)
csclassdata[:,1] = np.round(csclassdata[:,1],granularity)

# 将两张数据表按经纬度做联表操作
datacsDict = {} # 把 cloudsat数据转换成字典
for item in range(127):
    if item == 0:
        a = {'lon': csclassdata[:,0]}
    elif item == 1:
        a = {'lat':csclassdata[:,1]}
    else:
        a = {'type'+str(item-2):csclassdata[:,item]}
    datacsDict.update(a)
print(datacsDict)
datacs = pd.DataFrame(datacsDict)
# print(datacs)

datamodDict = {} # 把 modis数据转换成字典
for item in range(9):
    if item == 0:
        a = {'lon': modisdata[:,0]}
    elif item == 1:
        a = {'lat':modisdata[:,1]}
    else:
        a = {'bt'+str(item-2):modisdata[:,item]}
    datamodDict.update(a)
print(datamodDict)
datamod = pd.DataFrame(datamodDict)
# print(datamod)
conj_data = pd.merge(datamod,datacs,how = 'outer') # 联接两张数据表
print(conj_data)
matchedData = conj_data.dropna(axis=0,how='any') # 去掉含nan的行（未成功匹配的）
print(matchedData)

# exit()

# 保存数据
# dictkeys = matchedData.keys()
SAVEPATH = "E:\Code\python\cloudclassfication\data\MatchedData\\" + date + 'x3' + '.csv'
matchedData.to_csv(SAVEPATH,index=False,sep=',')

print("ok")