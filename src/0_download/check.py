import os

target_path='E:\Code\python\cloudclassfication\data\CloudSat'

days=os.listdir(target_path)
print('All content numbers is',len(days))

for day in days:
    # print(content.title(),target_path+'\\'+content)
    if os.path.isdir(target_path+'\\'+day):
        daysfiles = os.listdir(target_path+'\\'+day)

        # 验证每一天文件总数
        # if len(daysfiles) < 14:
        #     print(content.title(),len(daysfiles))

        # 验证文件大小
        for daysfile in daysfiles:
            sizex = os.path.getsize(target_path+'\\'+day+'\\'+daysfile) / 1024
            if sizex < 10000:
                print(day,daysfile,sizex)