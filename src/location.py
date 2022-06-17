from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *
import numpy as np
from HDFread import HDFread # 读取hdf文件
import pprint

def getLonAndLat(file):
    # file = "../data/UjNpYC_0001_0005/2008182071544_11562_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
    # hdf = SD(file)
    # cloudtype = hdf.select('CloudLayerType')
    longitude = HDFread(file,'Longitude')
    latitude = HDFread(file,'Latitude')
    longitude = np.array(longitude)
    latitude = np.array(latitude)
    # pprint.pprint(longitude)
    # pprint.pprint(latitude)
    # print('cloud type',cloudtype.get())
    # sst = list(HDFread(file,'SST','Data Fields'))
    return longitude,latitude



def segmentLat(file,start,end):
    # file = "../data/UjNpYC_0001_0005/2007051220514_04348_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"
    lon, lat = getLonAndLat(file)
    lat = np.array(lat.flatten())
    return np.all([lat > start, lat < end], axis=0)