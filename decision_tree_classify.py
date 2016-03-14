# coding=utf8
import numpy as np
import pandas as pd
from scipy import interp
from matplotlib import pyplot as plt
from sklearn import tree

import h5py 
fr = h5py.File(r'Data\Train\X.h5', 'r')
X = fr['X'].value
y = fr['y'].value.astype(bool)
w = fr['w'].value + 1

from train_predict import test
test(X, y, w)
