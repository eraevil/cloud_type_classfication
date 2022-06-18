import pandas as pd
import transbigdata as tbd
# 按文件名读取整个文件
cs = pd.read_csv("../../data/DoneData/CloudSat/CS20080101.csv")
mac = pd.read_csv('../../data/DoneData/MAC/MAC20080101.csv')
print(cs.head())
print(mac.head())


dfx = tbd.ckdnearest(cs,mac,Aname=['lon','lat'],Bname=['lon','lat'])
dfx.rename(columns={'index':'index_temp'},inplace=True)

d_rows=dfx[dfx['index_temp'].duplicated(keep=False)] # 找出同一Modis像元数据
dfx.drop(d_rows.index,axis=0,inplace=True) # 预删除
# 将重复的数据分组
# 分组后取均值，分别创建新的 row
g_items=d_rows.groupby('index_temp').mean()
g_items['index_temp']=g_items.index

# 将新的 row加入到原有数据并且重整
dfx=dfx.append(g_items)
dfx.sort_values(by='index_temp',inplace=True,ascending=False)
dfx.set_index(dfx['index_temp'],inplace=True)
# dfx.drop(['index'],axis=1,inplace=True)
dfx.to_csv('../../data/DoneData/20080101.csv',index=False)
# print(dfx)
print("ok!")