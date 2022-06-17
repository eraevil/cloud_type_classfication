from pyhdf.SD import *
from pyhdf.HDF import *
from pyhdf.VS import *
from pyhdf.V import  *
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pprint
import matplotlib.pyplot as plt
import netCDF4 as nc
import h5py

filepath = '2008182004012_11558_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf'
hdf = SD(filepath)
print(hdf.info())  # 信息类别数
# pprint.pprint(hdf.attributes())

# pprint.pprint(hdf.attributes())

# data = hdf.datasets()
# for i in data:
#     print(i)  # 具体类别

# height = hdf.select('Height')
# cloudscenario = hdf.select('cloud_scenario')
# cloudbase = hdf.select('CloudLayerBase')
# cloudtop = hdf.select('CloudLayerTop')
# cloudtype = hdf.select('CloudLayerType')

# no = 11728
# print('\n',no)
# print('cloud scenario',cloudscenario.get()[no][66])
# print('cloud type',cloudtype.get()[no])
# print('cloud top',cloudtop.get()[no])
# print('cloud base',cloudbase.get()[no])


# print(np.array(cloudtype.get()).shape)


# print(data.keys())
# hdf1 = HDF(filepath)
# print(hdf1.vstart())

# file = nc.Dataset(filepath)


import netCDF4 as nc

filename = "2008182004012_11558_CS_2B-CLDCLASS_GRANULE_P1_R05_E02_F00.hdf"

hdf = HDF(filename)
v = hdf.vgstart()
vs = hdf.vstart()
sd = SD(filename)

ref = v.find('Geolocation Fields')
print(ref)
vg = v.attach(ref)
members = vg.tagrefs()

nrecs = []
names = []
for tag, ref in members:
    # Vdata tag
    try:
        vd = vs.attach(ref)
    except:
        continue
    # nrecs, intmode, fields, size, name = vd.inquire()
    nrecs.append(vd.inquire()[0])  # number of records of the Vdata
    names.append(vd.inquire()[-1]) # name of the Vdata
    vd.detach()

print(nrecs)
print(names)

sd.end()
vs.end()
v.end()
# hdf.close()