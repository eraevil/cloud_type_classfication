

import wget
import subprocess
import os
import os.path
import shutil
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import csv

import time
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


def date2julian(date):
    year, month, day = int(date[0:4]),int(date[4:6]),int(date[6:8])
    hour = 10
    JD0 = int(365.25 * (year - 1)) + int(30.6001 * (1 + 13)) + 1 + hour / 24 + 1720981.5
    if month <= 2:
        JD2 = int(365.25 * (year - 1)) + int(30.6001 * (month + 13)) + day + hour / 24 + 1720981.5
    else:
        JD2 = int(365.25 * year) + int(30.6001 * (month + 1)) + day + hour / 24 + 1720981.5
    DOY = JD2 - JD0 + 1
    return int(DOY)

def match(day):
    # 匹配粒度
    granularity = 2  # 坐标保留小数位数
    # 数据文件列表
    date = day
    modisFilePath = fr'E:\Code\python\cloudclassfication\data\MODIS\{date}'
    modis_file_list = os.listdir(modisFilePath)
    csFilePath = fr'E:\Code\python\cloudclassfication\data\CloudSat\{date}'
    cs_file_list = os.listdir(csFilePath)

    # MODIS06数据获取
    modisdata = np.empty(shape=[0, 9], dtype=float)
    for path in modis_file_list:
        file = str(modisFilePath + '\\' + path)
        # file = "../../data/MODIS/MOD06_L2.A2008183.1455.061.2017292031620.hdf"
        hdf = SD(file)
        longitude = hdf.select('Longitude').get().flatten()  # 经度
        latitude = hdf.select('Latitude').get().flatten()  # 纬度
        modis = []
        modis = np.arange(len(longitude) * 9, dtype="float").reshape(len(longitude), 9)
        modis[:, 0] = longitude
        modis[:, 1] = latitude
        for i in range(7):
            bt = hdf.select('Brightness_Temperature').get()[i].flatten()  # 亮温数据
            modis[:, 2 + i] = bt
        # print(modis.shape)
        modisdata = np.append(modisdata, modis, axis=0)

    # CloudSat CLDCLASS数据获取
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

    # 调整坐标粒度
    modisdata[:, 0] = np.round(modisdata[:, 0], granularity)
    modisdata[:, 1] = np.round(modisdata[:, 1], granularity)
    csclassdata[:, 0] = np.round(csclassdata[:, 0], granularity)
    csclassdata[:, 1] = np.round(csclassdata[:, 1], granularity)

    # 将两张数据表按经纬度做联表操作
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
    # print(datacs)

    datamodDict = {}  # 把 modis数据转换成字典
    for item in range(9):
        if item == 0:
            a = {'lon': modisdata[:, 0]}
        elif item == 1:
            a = {'lat': modisdata[:, 1]}
        else:
            a = {'bt' + str(item - 2): modisdata[:, item]}
        datamodDict.update(a)
    # print(datamodDict)
    datamod = pd.DataFrame(datamodDict)
    # print(datamod)
    conj_data = pd.merge(datamod, datacs, how='outer')  # 联接两张数据表
    # print(conj_data)
    matchedData = conj_data.dropna(axis=0, how='any')  # 去掉含nan的行（未成功匹配的）
    # print(matchedData)

    # 保存数据
    # dictkeys = matchedData.keys()
    SAVEPATH = "E:\Code\python\cloudclassfication\data\MatchedData\\" + date + '.csv'
    matchedData.to_csv(SAVEPATH, index=False, sep=',')
    print(date,"ok")

def all_match(csv_path):
    df = np.array(pd.read_csv(csv_path))
    for index, item in enumerate(df):
        day = str(int(item[0]))
        if int(item[1]) == 1:
            print(day, "已匹配，跳过。")
            continue
        modisfile = fr'E:\Code\python\cloudclassfication\data\MODIS\{day}'
        cloudsatfile = fr'E:\Code\python\cloudclassfication\data\CloudSat\{day}'
        try:
            # print(index,day)
            if os.path.exists(modisfile) and os.path.exists(cloudsatfile):
                print("匹配",day)
                match(day)
            # 更新匹配标记
            df[index][1] = 1
            df1 = np.array(pd.read_csv(csv_path))
            df1[index][1] = 1
            pd.DataFrame(df1).to_csv(csv_path, index=False)
        except:
            print(item[0],"未成功")
if __name__ == '__main__':
    time_start = time.time()
    print("开始匹配")
    startday = '20080101'
    endday = '20081232'
    year = startday[0:4]
    startDoy = date2julian(startday)
    endDoy = date2julian(endday)
    date_list = pd.date_range(start=startday, periods=len(range(startDoy, endDoy))).strftime("%Y%m%d").tolist()
    doys = np.arange(startDoy, endDoy)
    csv_path = r'E:\Code\python\cloudclassfication\data\MatchedData\matcheddays.csv'
    flags = np.array(np.zeros((len(doys), 2)))
    flags[:, 0] = date_list
    if not os.path.exists(csv_path):
        pd.DataFrame(flags).to_csv(csv_path,index=False)
    # df = np.array(pd.read_csv(csv_path))
    all_match(csv_path)

    print("good job.")
    time_end = time.time()
    print('总计用时', round(time_end - time_start, 2), 's')