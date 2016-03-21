
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## load orginal data
df_train = pd.read_csv(r'Data/Train/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
df_test = pd.read_csv(r'Data/Test/PPD_Master_GBK_2_Test_Set.csv', encoding='gb18030')
df_label = df_train['target']
df_train.drop('target', axis=1, inplace=True)

## bind data
df_all = pd.concat([df_train,df_test])
# df_all.shape


# In[2]:

## 整理文字型数据
is_num_column = df_all.applymap(np.isreal).all(0) # Nan is also a np.isreal, type(Nan)=float64 !=np.nan
text_columns = is_num_column[is_num_column==False].index.values
# TODO should numerical these columns
# Show no text matrix
df_num = df_all.drop(text_columns,axis=1)
df_text = df_all.loc[:,text_columns]


# In[3]:

## 地址型数据整理
#['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_19', 'UserInfo_20', 'UserInfo_24']
## 排除3个地址型变量，只有以下四个变量合适
addr_column_names = ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_19']
df_address = df_all.loc[:, addr_column_names]
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
## 选取转化列
df_address = dd[['wyscore_x','wyscore_y']]
df_address.columns = ['UserInfo_2_wyscore','UserInfo_4_wyscore','UserInfo_7_wyscore','UserInfo_19_wyscore']


# In[4]:

## 文字型数据整理
## 比如Education_Info
## text category variables to dummys
textcat_column_names = ['UserInfo_9', 'UserInfo_22', 'UserInfo_23', 
                        'Education_Info2', 'Education_Info3', 'Education_Info4', 'Education_Info6', 'Education_Info7', 'Education_Info8', 
                        'WeblogInfo_19', 'WeblogInfo_20', 'WeblogInfo_21']
df_otext = df_text.loc[:,textcat_column_names]
df_otext = pd.get_dummies(df_otext.loc[:, textcat_column_names])


# In[5]:

## 将地址特征与dummys特征和原数据融合
df_all1 = pd.concat([df_num.reset_index(drop=True), df_otext.reset_index(drop=True), df_address.reset_index(drop=True)], axis=1)
## 将train和test数据分开
df_train_feature = df_all1.loc[df_all1.Idx.isin(df_train.Idx),]
df_test_feature = df_all1.loc[df_all1.Idx.isin(df_test.Idx),]
## unit test
## df_test_feature.shape[0] + df_train_feature.shape[0] == df_all1.shape[0]


# In[7]:

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


# In[8]:

import h5py
## Train
fw = h5py.File(r'Data/Train/X-Text.h5', 'w')
fw.create_dataset('X', data=X1)
fw.create_dataset('y', data=df_label)
fw.create_dataset('w', data=weight)
fw.close()
## Test
fw = h5py.File(r'Data/Test/X-Text.h5', 'w')
fw.create_dataset('X', data=X2)
fw.close()


# In[127]:

## 合并整理数据，有问题，train和test应该一起变换
# import util
# #util.numericalization(df_all)
# util.add_missing_column_imputation(df_train_feature)
# X2 = util.categorical_binarization(df_train_feature)
# print 'transform finished, saving...'
# weight = df_label * 20 + 1

# import h5py
# fw = h5py.File(r'Data/Train/X.h5', 'w')
# fw.create_dataset('X', data=X2)
# fw.create_dataset('y', data=df_label)
# fw.create_dataset('w', data=weight)
# fw.close()

# ## test set
# ## 逸轩数值型数据整理部分
# import util
# #util.numericalization(df_all)
# util.add_missing_column_imputation(df_test_feature)
# X2 = util.categorical_binarization(df_test_feature)
# print 'transform finished, saving...'
# weight = df_label * 20 + 1

# import h5py
# fw = h5py.File(r'Data/Test/X.h5', 'w')
# fw.create_dataset('X', data=X2)
# fw.close()


# In[8]:




# In[ ]:

# ## For R
# df_train_feature_wlabel = pd.concat([df_train_feature,df_label],axis=1)
# df_train_feature_wlabel.to_csv('df_train_feature_wlabel.csv',index=False,encoding='utf-8')

