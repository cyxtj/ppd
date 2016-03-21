# coding=utf8
import sys
import numpy as np
import h5py
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # fr = h5py.File(r'Data\Train\X.h5', 'r')
    fr = h5py.File(r'Data/Train/NewMaster.h5', 'r')
    pca = PCA(n_components=int(sys.argv[1]))
    x = pca.fit_transform(fr['X'].value)
    y = fr['y'].value
    w = fr['w'].value
        
    #output
    fr.close()
    fw = h5py.File(r'Data/Train/NewMaster-diam-reduc.h5', 'w')
    fw.create_dataset('X', data=x)
    fw.create_dataset('y', data=y)
    fw.create_dataset('w', data=w)
    fw.close()
    
