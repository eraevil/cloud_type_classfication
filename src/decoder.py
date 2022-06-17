import numpy as np

def dec(num):
    """输入一个含云类型编码信息的10进制数，二进制解码，转换为10进制的云类型0-9"""
    numbin = '{:16b}'.format(num)
    numbin = numbin.replace(' ','0')
    type = int(numbin[-5:-1],2)
    print(num,numbin,numbin[-5:-1],type)
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