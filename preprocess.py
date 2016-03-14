import pandas as pd

X = pd.read_csv(r'Data/Train/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
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
fw = h5py.File(r'Data/Train/X.h5', 'w')
fw.create_dataset('X', data=X2)
fw.create_dataset('y', data=y)
fw.create_dataset('w', data=weight)
fw.close()

