#!/usr/bin/python
# -*- coding: utf-8 -*-

projectroot = "D:\project\cloud_type_classfication"


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

def download(year,dest,csv_path):
    df = np.array(pd.read_csv(csv_path))
    index = 1
    for index,item in enumerate(df):
        print(index,int(item[2]))
        if int(item[2]) == 1:
            index += 1
            print(str(int(item[1])),"已下载，跳过。")
            continue

        # 下载
        path = dest + '\\' + str(int(item[1]))
        if not os.path.exists(path):
            os.mkdir(path)
        print("下载",str(int(item[1])),"数据")
        cmd = "sftp lisheng_rcAT163.com@www.cloudsat.cira.colostate.edu:Data/2B-CLDCLASS.P1_R05/" + year + "/" + str(int(item[0])).zfill(3) + "/* " + path
        # print(cmd)
        os.system(cmd)

        # 更新下载标记
        df[index][2] = 1
        df1 = np.array(pd.read_csv(csv_path))
        df1[index][2] = 1
        pd.DataFrame(df1).to_csv(csv_path,index=False)
        index += 1
        print(str(int(item[1])),"下载完毕")
    print("全部下载完毕")

if __name__ == '__main__':
    print("开始下载Cloudsat数据")
    startday = '20090101'
    endday = '20091232'
    year = startday[0:4]
    startDoy = date2julian(startday)
    endDoy = date2julian(endday)
    date_list = pd.date_range(start=startday, periods=len(range(startDoy, endDoy))).strftime("%Y%m%d").tolist()
    doys = np.arange(startDoy,endDoy)
    print(date_list)
    print(doys)
    flags = np.array(np.zeros((len(doys),3)))
    flags[:,0] = doys
    flags[:,1] = date_list
    csv_path = projectroot + r'\data\CloudSat\flags.csv'
    dest = projectroot + r'\data\CloudSat'
    if not os.path.exists(csv_path):
        pd.DataFrame(flags).to_csv(csv_path,index=False)
    df = np.array(pd.read_csv(csv_path))

    download(year,dest,csv_path)
    # cmd = "sftp lisheng_rcAT163.com@www.cloudsat.cira.colostate.edu:Data/2B-CLDCLASS.P1_R05/2008/001/* E:\Code\python\cloudclassfication\data\CloudSat\\test"
    # os.system(cmd)


    # download(dest,csv_path)
    # os.system("wget -nd -N -r --user=hehongjian --password=!r3XmH ftp://ftp.cloudsat.cira.colostate.edu/2B-CLDCLASS.P1_R05/2008/001 -P E:\Code\python\cloudclassfication\data\CloudSat\\test")

    print("well done!")