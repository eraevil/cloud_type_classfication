import pandas as pd
import transbigdata as tbd
import warnings;
warnings.filterwarnings("ignore")




def DAYlist(start,end):
    yearmonthday = pd.date_range(start,end,freq="D").strftime("%Y%m%d").to_list()
    return yearmonthday

days = DAYlist('2008-01-01','2008-12-31')
index = 0
for day in days:
    print(day)
    # 按文件名读取整个文件
    try:
        cs_path = "../../data/DoneData/CloudSat/CS" + day + ".csv"
        mac_path = "../../data/DoneData/MAC/MAC" + day + ".csv"
        cs = pd.read_csv(cs_path)
        mac = pd.read_csv(mac_path)
        # print(cs.head())
        # print(mac.head())

        dfx = tbd.ckdnearest(cs,mac,Aname=['lon','lat'],Bname=['lon','lat'])
        dfx.rename(columns={'index':'index_temp'},inplace=True)
        dfx['top'] = dfx['top'] * 1000
        dfx['cth'] = dfx['cth'].replace(-32767,-99000)
            # where(dfx['cth'] == '-32767', '-99000',dfx['cth'])

        d_rows=dfx[dfx['index_temp'].duplicated(keep=False)] # 找出同一Modis像元数据
        # dfx.drop(d_rows.index,axis=0,inplace=True) # 预删除
        # 将重复的数据分组
        # 分组后取均值，分别创建新的 row
        g_items=d_rows.groupby('index_temp').mean()
        g_items['index_temp']=g_items.index

        # 将新的 row加入到原有数据并且重整
        dfx=dfx.append(g_items)
        # dfx = pd.concat(g_items)
        dfx.sort_values(by='index_temp',inplace=True,ascending=False)
        # dfx.set_index(dfx['index_temp'],inplace=True)
        dfx.drop(['index_temp'],axis=1,inplace=True) # 删除列
        dfx.drop(['lon_x'],axis=1,inplace=True) # 删除列
        dfx.drop(['lat_x'],axis=1,inplace=True) # 删除列
        dfx.drop(['dist'],axis=1,inplace=True) # 删除列
        # dfx.drop(['index'],axis=1,inplace=True)
        # dfx.to_csv('../../data/DoneData/20080101.csv',index=False)

        if index == 0:
            dfx.to_csv('../../data/DoneData/2008.csv', mode='a', index=False)
        else:
            dfx.to_csv('../../data/DoneData/2008.csv', mode='a', index=False, header=False)
        index += 1
    except Exception as e:
        print("出现了错误")
        print('错误明细是', e.__class__.__name__, e)  # continue#jia
        continue

    print("ok")

# print(dfx)
print("完毕！")