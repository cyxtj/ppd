import numpy as np
import pandas as pd
'''
filename = r'PPD_Training_Master_GBK_3_1_Training_Set'
filename = 'NewMaster'
X = pd.read_csv(r'Data/Train/%s.csv'%filename, encoding='gbk')
# type_desc = pd.read_csv(r'Data\column_descrip.csv')
print 'data loaded, transforming...'
y = X['target']
X.drop('target', axis=1, inplace=True)
# is_categorical = type_desc['type']=='Categorical'
# X_categorical_columns = type_desc[is_categorical]['name']
# X_numerical_columns = type_desc[is_categorical]['name']
# X_cate = X[X_categorical_columns].copy()
# X_num = X[X_numerical_columns].copy()
# # For categorical data:
# X_cate.applymap(np.isreal).all(1)
# X_num.applymap(np.isreal).all(1)

import util
util.numericalization(X)
util.add_missing_column_imputation(X)
X2 = util.categorical_binarization(X)
print 'transform finished, saving...'
weight = y * 20 + 1

import h5py
fw = h5py.File(r'Data/Train/%s.h5'%filename, 'w')
fw.create_dataset('X', data=X2)
fw.create_dataset('y', data=y)
fw.create_dataset('w', data=weight)
fw.close()
'''

import h5py 
fr = h5py.File(r'Data/Train/X-Text.h5', 'r')
X = fr['X'].value
y = fr['y'].value.astype(bool)
w = fr['w'].value
print 'Text features loaded'
print 'X.shape = ', X.shape

# update
updateInfo = pd.read_csv(r'Data/Train/Update.csv').values
# period diff
periodDiff = pd.read_csv(r'Data/Train/PeriodDiff.csv').values
print 'update and periodDiff loaded'

# join
X = np.hstack([X, updateInfo, periodDiff])

fw = h5py.File(r'Data/Train/X-Text-Update-PeriodDiff.h5', 'w')
fw.create_dataset('X', data=X)
fw.create_dataset('y', data=y)
fw.create_dataset('w', data=w)
fw.close()

## test data
fr = h5py.File(r'Data/Test/X-Text.h5', 'r')
X = fr['X'].value
print 'Text features loaded'
print 'X.shape = ', X.shape

# update
updateInfo = pd.read_csv(r'Data/Test/Update.csv').values
# period diff
periodDiff = pd.read_csv(r'Data/Test/PeriodDiff.csv').values
print 'update and periodDiff loaded'

# join
X = np.hstack([X, updateInfo, periodDiff])

import h5py
fw = h5py.File(r'Data/Test/X-Text-Update-PeriodDiff.h5', 'w')
fw.create_dataset('X', data=X)
fw.close()
