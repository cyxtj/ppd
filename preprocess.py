import numpy as np
import pandas as pd


import h5py 
def concate_features(filelist, save_name):
    features = []
    for f in filelist:
        df = pd.read_csv('Data/Train/'+f)
        print '%i features in %s'%(df.shape[1], f)
        features.append(df)
    X = pd.concat(features)
    y = pd.read_csv('Data/Train/y.csv')
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
    X = pd.concat(features)

    fw = h5py.File(r'Data/Test/'+save_name, 'w')
    fw.create_dataset('X', data=X)
    fw.close()


if __name__ == '__main__':
    ## merge many features together
    # requires X-Text, deal with text information
    import feature_bind_fixed
    filelist = [
            'X-Text.csv', 
            'Update.csv', 
            'PeriodDiff.csv', 
            'update_summary.csv'
            ]
    concate_features(filelist, 'Xfeatures.h5')
