import numpy as np
from pyhdf.HDF import *
from pyhdf.V import *
from pyhdf.VS import *
from pyhdf.SD import *

def decoderScenario(filepath):
    hdf = SD(filepath)
    cloudscenario = hdf.select('cloud_scenario')
    cloudscenarioV = cloudscenario.get()
    cloudscenarioV_set = set(cloudscenarioV.flatten()) # 拉伸数组，提取云类型编码值表
    typeDict = scenariolist2type(cloudscenarioV_set) # 得到云类型编码字典
    # 通过字典对云类型进行解码
    for i in typeDict:
        cloudscenarioV[cloudscenarioV == i] = typeDict[i]
    # print(cloudscenarioV)
    return cloudscenarioV

def dec(num):
    """输入一个含云类型编码信息的10进制数，二进制解码，转换为10进制的云类型0-9"""
    numbin = '{:16b}'.format(num)
    numbin = numbin.replace(' ','0')
    type = int(numbin[-5:-1],2)
    # print(num,numbin,numbin[-5:-1],type)
    # type = int(numbin[-5:-1], 2)
    # print("dec", num)
    # print("bin", numbin)
    # print("type", type)
    return type

def scenariolist2type(scenarioArr):
    typeDict = {}
    for item in scenarioArr:
        a = {item: dec(item)}
        typeDict.update(a)
    return typeDict



# dec(18785)