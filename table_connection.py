#coding=utf8
import numpy as np
import pandas as pd

# 分析用户Update数据，看用户都修改了哪些数据
# 与Master表中的UserInfo对比，尝试分析UserInfo的属性
# 发现只有29995个Idx
# 未完成
UU = pd.read_csv(r'Data\Train\PPD_Userupdate_Info_3_1_Training_Set.csv')
gUU = UU.groupby('Idx')

unique_cate = UU['UserupdateInfo1'].unique()
cate_noempty = {unique_cate[i]: i+1 for i in range(unique_cate.shape[0])}

INFO = np.zeros((gUU.ngroups, unique_cate.shape[0]+1))
for i, [name, group] in enumerate(gUU):
    update_cate = group['UserupdateInfo1'].values
    info_one_record = np.zeros(unique_cate.shape[0]+1)
    info_one_record[0] = name
    for cate in update_cate:
        info_one_record[cate_noempty[cate]] = 1
    INFO[i, :] = info_one_record

cates = unique_cate.tolist()
cates.insert(0, 'Idx')
df = pd.DataFrame(INFO, columns=cates)
df.to_csv(r'Data\Train\a_Userupdate.csv')


