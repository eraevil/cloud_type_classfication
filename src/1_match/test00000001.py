import numpy as np
import pandas as pd


x = np.arange(9).reshape(3,3) / 1.0
y = np.arange(9).reshape(3,3) / 2
print(x)
print(y)
# print(x[:,0])
# pos = np.where(x[:,0]>2,1,0)
# print(pos)

b = [1,11,1,4,3,2,0,1,2,3,5,6,7]
b=set(b)
print(b)
df = pd.DataFrame(x).groupby(1, as_index=False)
print(df)