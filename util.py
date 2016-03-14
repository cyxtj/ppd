#coding=utf8
import numpy as np
import pandas as pd

from sklearn import preprocessing

def numericalization(X):
    addr_column_names = ['UserInfo_2', 'UserInfo_4', 'UserInfo_7', 'UserInfo_8', 'UserInfo_19', 'UserInfo_20', 'UserInfo_24']
    X.drop(addr_column_names, axis=1, inplace=True)
    X.drop('ListingInfo', axis=1, inplace=True)
    is_num_column = X.applymap(np.isreal).all(0) # Nan is also a np.isreal, type(Nan)=float64 !=np.nan
    text_columns = is_num_column[is_num_column==False].index.values
    # TODO should numerical these columns
    X.drop(text_columns, axis=1, inplace=True)
    # X2 = X[is_num_column[is_num_column==True].index.values]


def add_missing_column_imputation(X):
    '''
    X only contains numbers
    Replace -1 with Nan with stand for missing values.
    For each column in X that contains missing values, create a new column: #_miss
    where 1 means miss and 0 means fill
    '''
    X.replace(-1, np.nan, inplace=True)
    miss_table = X.isnull()
    has_miss_column = miss_table.any(0)
    miss_columns = has_miss_column[has_miss_column==True].index.values
    for c in miss_columns:
        X[c+'_miss'] = miss_table[c].astype(int)
    # TODO now simply set missing value as 0
    X.fillna(value=0, inplace=True)
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

def categorical_binarization(X):
    '''
    X only contains numbers, and all entries are filled.
    Define a column of data is categorical: 
        if the number of distinct values in that column is less than 6
    binarize these categorical columns
    '''
    # find categorical columns
    to_binarize = np.zeros(X.shape[1])
    for i, column_name in enumerate(X):
        if '_miss' in column_name: continue     # this column is binary
        distinct_values = X[column_name].unique()
        nd = distinct_values.shape[0]
        if nd < 6 and nd > 2: # only two value don't need binarize
            print nd
            to_binarize[i] = 1
    enc = preprocessing.OneHotEncoder(categorical_features=to_binarize.astype(bool), sparse=False)
    Z = enc.fit_transform(X)
    return Z

def equal_minus1(x):
    return x==-1

