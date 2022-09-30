# cloud_type_classfication
Based on deep-learing classifying cloud type

> 功能模块
- [X] 数据下载
- [X] 数据匹配，形成数据集
- [X] 训练
- [ ] 调参优化


> 数据匹配方案
按计算cloudsat廓线点到modis像元的距离，将cloudsat廓线点标记为最近modis像元所属点，计算同一modis像元的clousat云类型均值作为与该像元匹配的数据。

> 数据说明
- Cloudsat卫星云分类产品数据

CloudSat 卫星也是美国地球观测系统 EOS 中的太阳同步极轨卫星，于 2006 年 4 月 28 日成功发射，和其他 4 颗卫星(Aqua、Aura、CALIPSO 和 PARASOL)组成  “A-Train” 卫星编队。它的轨道高度是 705km，倾角是 98.2°，卫星绕地球一周被称作一个扫描轨道，一个轨道的扫描时间是 99min 左右。

![image](https://user-images.githubusercontent.com/35321279/178888589-6fd089bd-728b-436c-b58b-5076f4382fb8.png)


![image](https://user-images.githubusercontent.com/35321279/178888432-41f4f589-2b4c-4e00-b3b0-f446c62b8cfe.png)

每天大约有 14、15 个扫描轨道，扫描长度是40022km 左右，每个轨道有 36383 个星下像素点（扫描廓线点）。

![image](https://user-images.githubusercontent.com/35321279/179341797-bbb875f0-6f6e-43b9-904c-dc88afb5570e.png)

![image](https://user-images.githubusercontent.com/35321279/178888518-bfdfbcce-9656-4573-9452-6298fe048673.png)

每个垂直剖面每隔 240m 就会获得一个扫描数据，从高空到地面一共可以获得 125 个不同高度上的数据，这也就是说它的垂直探测的高度是 30km 左右。

![image](https://user-images.githubusercontent.com/35321279/179341756-df72bb00-f633-4601-94ea-8aa06a0bdce7.png)

通过对cloudsat云类型数据解码可以得到各个廓线点，垂直高度上125层的云类型数据，将该数据作为该点的labels，后续进行模型学习。

- MODIS卫星产品数据

MOD06S0  

MAC06S0

使用该数据集的1km分辨率数据，包含cpi、ctp、cth、ctt、sft、ce、ce11等云光学及物理性质

MAC03S0

包含与MAC06S0数据集1km分辨率重合的地理信息lon、lat

- CLIPSO卫星产品数据

包含云分类信息，需要解码
