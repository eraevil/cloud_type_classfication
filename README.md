# cloud_type_classfication
Based on deep-learing classifying cloud type

> 功能模块
- [X] 数据下载
- [X] 数据匹配，形成数据集
- [ ] 训练
- [ ] 调参优化

> 数据匹配方案
按计算cloudsat廓线点到modis像元的距离，将cloudsat廓线点标记为最近modis像元所属点，计算同一modis像元的clousat云类型均值作为与该像元匹配的数据。

