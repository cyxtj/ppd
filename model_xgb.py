# coding=utf8
# this is the quick implement of xgboosting
# should embed into main.py soon
import numpy as np
import pandas as pd
from scipy import interp
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn import tree

import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import h5py 
# fr = h5py.File(r'Data\Train\X.h5', 'r')
fr = h5py.File(r'Data/Train/X.h5', 'r')
X = fr['X'].value
y = fr['y'].value.astype(bool)
w = fr['w'].value

import xgboost as xgb


from sklearn.cross_validation import cross_val_score, StratifiedKFold
skf = StratifiedKFold(y, n_folds=8, shuffle=True)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
for i, (train_index, test_index) in enumerate(skf):
    print i, 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w_train, w_test = w[train_index], w[test_index]
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    param = {'silent': 1, 'max_depth':1}
    bst = xgb.train(param, dtrain, num_boost_round=50)
    dtest = xgb.DMatrix(X_test)
    p = bst.predict(dtest)
    fpr, tpr, thresholds = roc_curve(y_test, p)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.3f)' % (i, roc_auc))
    print 'test auc: %0.3f'%roc_auc, 
    p2 = bst.predict(dtrain)
    fpr, tpr, thresholds = roc_curve(y_train, p2)
    roc_auc = auc(fpr, tpr)
    print 'train auc: %0.3f'%roc_auc

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= skf.n_folds
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.3f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

