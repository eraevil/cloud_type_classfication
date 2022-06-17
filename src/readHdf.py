from pyhdf.SD import SD
import matplotlib.pyplot as plt
import numpy as np
import pprint
import matplotlib.pyplot as plt

hdf = SD('../data/ISCCP.D2.0.GLOBAL.2008.04.99.9999.GPC.hdf')
print(hdf.info())  # 信息类别数


data = hdf.datasets()
for i in data:
    print(i)  # 具体类别

set6 = hdf.select('Data-Set-22')
print(set6)

pprint.pprint(set6.attributes())
pprint.pprint(set6[0])

attr = set6.attributes(full=1)
attNames = attr.keys()
pprint.pprint(attNames)


data = set6[:]
x = []
y = []
value = []
rows = len(data)
cols = len(data[0])

for i in range(rows):
    for j in range(cols):
        x.append(i)
        y.append(j)
        value.append(data[i][j])

# print(rows,cols,value)
plt.scatter(x,y,0.01,value)
plt.colorbar()
plt.show()

exit('0812')