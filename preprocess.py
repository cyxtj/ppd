import numpy as np
import pandas as pd


import h5py 
def concate_features(filelist, save_name):
    ## merge many features together
    features = []
    for f in filelist:
        df = pd.read_csv('Data/Train/'+f)
        print '%i features in %s'%(df.shape[1], f)
        features.append(df)
    X = pd.concat(features, axis=1)
    y = pd.read_csv('Data/Train/y.csv', header=None)
    w = y * 20 + 1

    fw = h5py.File(r'Data/Train/'+save_name, 'w')
    fw.create_dataset('X', data=X)
    fw.create_dataset('y', data=y)
    fw.create_dataset('w', data=w)
    fw.close()

    features = []
    for f in filelist:
        df = pd.read_csv('Data/Test/'+f)
        print '%i features in %s'%(df.shape[1], f)
        features.append(df)
    X = pd.concat(features, axis=1)

    fw = h5py.File(r'Data/Test/'+save_name, 'w')
    fw.create_dataset('X', data=X)
    fw.close()

def process_master():
    ## load orginal data
    df_train = pd.read_csv(r'./Data/Train/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
    df_test = pd.read_csv(r'./Data/Test/PPD_Master_GBK_2_Test_Set.csv', encoding='gb18030')
    df_label = df_train['target']
    df_train.drop('target', axis=1, inplace=True)
    df_all = pd.concat([df_train,df_test])
    import util
    #util.numericalization(df_all)
    util.numericalization(df_all)
    util.add_missing_column_imputation(df_all)
    X_all = util.categorical_binarization(df_all)
    print X_all
    ## train
    X1 = X_all.iloc[:30000, :]
    weight = df_label * 20 + 1
    ## test
    X2 = X_all.iloc[30000:, :]
    X1.to_csv(r'Data/Train/Master.csv', index=False, encoding='utf8')
    df_label.to_csv(r'Data/Train/y.csv', index=False)
    ## Test
    X2.to_csv(r'Data/Test/Master.csv', index=False, encoding='utf8')

    '''
    import h5py
    ## Train
    fw = h5py.File(r'Data/Train/Xfeatures1.h5', 'w')
    fw.create_dataset('X', data=X1)
    fw.create_dataset('y', data=df_label)
    fw.create_dataset('w', data=weight)
    fw.close()
    ## Test
    fw = h5py.File(r'Data/Test/Xfeatures1.h5', 'w')
    fw.create_dataset('X', data=X2)
    fw.close()
    '''





if __name__ == '__main__':
    # process_master()
    import feature_filter_category
    filelist = [
            'Master.csv', 
            'Update_Y.csv', 
            'Combination_Z.csv', 
            ]
    concate_features(filelist, 'Xfeatures.h5')
