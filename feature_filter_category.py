
# coding: utf-8

# In[48]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load orginal data
df_train = pd.read_csv(r'./Data/Train/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
df_test = pd.read_csv(r'./Data/Test/PPD_Master_GBK_2_Test_Set.csv', encoding='gb18030')
df_label = df_train['target']
df_train.drop('target', axis=1, inplace=True)

## bind data
df_all = pd.concat([df_train,df_test])
# df_all.shape


# In[49]:

## 整理文字型数据
is_num_column = df_all.applymap(np.isreal).all(0) # Nan is also a np.isreal, type(Nan)=float64 !=np.nan
text_columns = is_num_column[is_num_column==False].index.values
# TODO should numerical these columns
# Show no text matrix
df_num = df_all.drop(text_columns,axis=1)
df_text = df_all.loc[:,text_columns]


# In[50]:

## 地址型数据整理
#['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_19', 'UserInfo_20', 'UserInfo_24']
## 排除3个地址型变量，只有以下四个变量合适
addr_column_names = ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_19']
df_address = df_all.loc[:, addr_column_names]
## 统一格式
df_address.UserInfo_7 = df_address.UserInfo_7 + u'省'
df_address.UserInfo_2 = df_address.UserInfo_2 + u'市'
df_address.UserInfo_4 = df_address.UserInfo_4 + u'市'
## 针对这四个变量设置映射函数，需加载训练数据中的对应关系
## 读取数据映射
df_ui2 = pd.read_csv('Data/UserInfo_2_feature_map.csv')
df_ui4 = pd.read_csv('Data/UserInfo_4_feature_map.csv')
df_ui7 = pd.read_csv('Data/UserInfo_7_feature_map.csv')
df_ui19 = pd.read_csv('Data/UserInfo_19_feature_map.csv')
## 和映射表融合
dd = df_address.merge(df_ui2, left_on='UserInfo_2', right_on='name', how='left').fillna(0)
dd = dd.merge(df_ui4, left_on='UserInfo_4', right_on='name', how='left').fillna(0)
dd = dd.merge(df_ui7, left_on='UserInfo_7', right_on='name', how='left').fillna(0)
dd = dd.merge(df_ui19, left_on='UserInfo_19', right_on='name', how='left').fillna(0)
df_address.head()


# In[51]:

## 转化经纬度
df_loctable = pd.read_csv('Data/city_lonlat.csv', encoding='gbk')
df_loctable.head()
## 读取省会信息以便把省份经纬度转化为省会的经纬度
df_capital = pd.read_csv('Data/capital_name.csv', encoding='gbk')
df_capital.iloc[26,0] = u'新疆维吾尔' # 结合ppd格式进行调整
df_capital.iloc[22,0] = u'港澳台' # 结合ppd格式进行调整,台湾
df_capital.iloc[28,0] = u'港澳台' # 结合ppd格式进行调整,澳门
df_capital.iloc[31,0] = u'港澳台' # 结合ppd格式进行调整,香港
df_capital.iloc[:,2] = df_capital.iloc[:,2] + u'市'
## 获取省会经纬度
df_capital = df_capital.merge(df_loctable, left_on=u'省会', right_on=u'区县', how='left').fillna(0)
## 经纬度特征转化
df_lonlat = df_address.merge(df_loctable, left_on='UserInfo_2', right_on=u'区县', how='left').fillna(0)
df_lonlat = df_lonlat.drop(['UserInfo_2', u'省份',u'地市',u'区县'],axis=1)
df_lonlat = df_lonlat.merge(df_loctable, left_on='UserInfo_4', right_on=u'区县', how='left').fillna(0)
df_lonlat = df_lonlat.drop(['UserInfo_4', u'省份',u'地市',u'区县'],axis=1)
df_lonlat = df_lonlat.merge(df_capital, left_on='UserInfo_7', right_on=u'省份_x', how='left').fillna(0)
df_lonlat = df_lonlat.drop(['UserInfo_7', u'省份_x',u'简称',u'省会',u'省份_y',u'地市',u'区县'],axis=1)
df_lonlat = df_lonlat.merge(df_capital, left_on='UserInfo_19', right_on=u'省份_x', how='left').fillna(0)
df_lonlat = df_lonlat.drop(['UserInfo_19', u'省份_x',u'简称',u'省会',u'省份_y',u'地市',u'区县'],axis=1)
## 命名特征
df_lonlat.columns = ['ui2_lon','ui2_lat','ui4_lon','ui4_lat','ui7_lon','ui7_lat','ui19_lon','ui19_lat']


# In[52]:

## 选取转化列
df_address = dd[['wyscore_x','wyscore_y']]
df_address.columns = ['UserInfo_2_wyscore','UserInfo_4_wyscore','UserInfo_7_wyscore','UserInfo_19_wyscore']


# In[53]:

## 文字型数据整理
## 比如Education_Info
## text category variables to dummys
textcat_column_names = ['UserInfo_9', 'UserInfo_22', 'UserInfo_23', 
                        'Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6', 'Education_Info7', 'Education_Info8', 
                        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21']
df_otext = df_text.loc[:,textcat_column_names]
df_otext = pd.get_dummies(df_otext.loc[:, textcat_column_names])


# In[54]:

## 将地址特征与dummys特征和原数据融合
df_all1 = pd.concat([df_num.reset_index(drop=True), df_otext.reset_index(drop=True),                      df_address.reset_index(drop=True), df_lonlat.reset_index(drop=True)], axis=1)
## 将train和test数据分开
df_train_feature = df_all1.loc[df_all1.Idx.isin(df_train.Idx),]
df_test_feature = df_all1.loc[df_all1.Idx.isin(df_test.Idx),]
## unit test
#df_test_feature.shape[0] + df_train_feature.shape[0] == df_all1.shape[0]


# In[145]:

## train和test一起变化
import util
#util.numericalization(df_all)
util.add_missing_column_imputation(df_all1)
X_all = util.categorical_binarization(df_all1)
## train
X1 = X_all[0:30000] #train
weight = df_label * 20 + 1
## test
X2 = X_all[30000:49999]
# X1.shape
# X2.shape


# In[124]:

## 20160328更改category函数，返回dataframe
X_all.head()


# In[125]:

## 3.a
## 筛选出train 和 test中差别比较大的几个特征
## train每列的均值
train_mean = np.mean(X1,0)
## train每列的标准差
train_std = np.std(X1,0)


# In[126]:

## test每列的均值
test_mean = np.mean(X2,0)
## test每列的标准差
test_std = np.std(X2,0)


# In[127]:

## 筛选原理
## 筛选mean异常的特征
max((train_mean - test_mean)/ train_mean)
train_mean_per = (train_mean - test_mean)/train_mean
train_mean_per = np.abs(train_mean_per)
## 查看mean的分布
sum(train_mean_per[np.argsort(train_mean_per)[::-1]]>0.5)


# In[128]:

## 选取应该删去的特征
delete_mean = np.argsort(train_mean_per)[::-1][0:68]


# In[129]:

## 筛选std异常的特征
max((train_std - test_std)/ train_std)
train_std_per = (train_std - test_std)/train_std
train_std_per = np.abs(train_std_per)
## 查看mean的分布
sum(train_std_per[np.argsort(train_std_per)[::-1]]>0.5)


# In[130]:

## 选取应该删去的特征
delete_std = np.argsort(train_std_per)[::-1][0:43]


# In[131]:

## 合并所有应该剔除的特征
delete_feature = np.unique(np.append(delete_mean,delete_std))


# In[143]:

## 获取需要删除的特征名单
delete_feature_name = np.argsort(train_mean_per)[::-1].index[delete_feature]
#delete_feature_name = pd.Series(delete_feature_name)
#delete_feature_name = delete_feature_name.astype(basestring)


# In[134]:

## 剔除train和test的这些特征
X1.drop(delete_feature_name,axis=1, inplace=True)
X2.drop(delete_feature_name,axis=1, inplace=True)


# In[217]:

## 导出X3,X4
# import h5py
# ## Train
# fw = h5py.File(r'Data/Train/X3.h5', 'w')
# fw.create_dataset('X', data=X3)
# fw.create_dataset('y', data=df_label)
# fw.create_dataset('w', data=weight)
# fw.close()
# ## Test
# fw = h5py.File(r'Data/Test/X4.h5', 'w')
# fw.create_dataset('X', data=X4)
# fw.close()

print X1.shape
print 'writin to files... '
## Train
X1.to_csv(r'Data/Train/Master.csv', index=False, encoding='utf8')
df_label.to_csv(r'Data/Train/y.csv', index=False)
## Test
X2.to_csv(r'Data/Test/Master.csv', index=False, encoding='utf8')
