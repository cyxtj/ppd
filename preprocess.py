import numpy as np
import pandas as pd

'''
filename = r'PPD_Training_Master_GBK_3_1_Training_Set'
# filename = 'NewMaster'
X = pd.read_csv(r'Data/Train/%s.csv'%filename, encoding='gbk')
print 'data loaded, transforming...'
y = X['target']
X.drop('target', axis=1, inplace=True)

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

## merge many features together
# requires X-Text, deal with text information
import feature_bind_fixed

import h5py 
fr = h5py.File(r'Data/Train/X-Text.h5', 'r')
X = fr['X'].value
y = fr['y'].value.astype(bool)
w = fr['w'].value
updateInfo = pd.read_csv(r'Data/Train/Update.csv').values
periodDiff = pd.read_csv(r'Data/Train/PeriodDiff.csv').values
updateSum = pd.read_csv(r'Data/Train/update_summary.csv').values
print 'X-Text.shape = ', X.shape
print 'updateInfo.shape: ', updateInfo.shape
print 'periodDiff.shape: ', periodDiff.shape
print 'updateSum.shape: ', updateSum.shape

# join
X = np.hstack([X, updateInfo, periodDiff, updateSum])
print 'X-train.shape: ', X.shape

fw = h5py.File(r'Data/Train/X-Text-Update-PeriodDiff.h5', 'w')
fw.create_dataset('X', data=X)
fw.create_dataset('y', data=y)
fw.create_dataset('w', data=w)
fw.close()

print '================================='

fr = h5py.File(r'Data/Test/X-Text.h5', 'r')
X = fr['X'].value
updateInfo = pd.read_csv(r'Data/Test/Update.csv').values
periodDiff = pd.read_csv(r'Data/Test/PeriodDiff.csv').values
updateSum = pd.read_csv(r'Data/Test/update_summary.csv').values
print 'X-Text.shape = ', X.shape
print 'updateInfo.shape: ', updateInfo.shape
print 'periodDiff.shape: ', periodDiff.shape
print 'updateSum.shape: ', updateSum.shape

# join
X = np.hstack([X, updateInfo, periodDiff, updateSum])
print 'X-test.shape: ', X.shape

import h5py
fw = h5py.File(r'Data/Test/X-Text-Update-PeriodDiff.h5', 'w')
fw.create_dataset('X', data=X)
fw.close()
