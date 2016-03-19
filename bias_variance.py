# coding=utf8
# this script do some work on parameter tuning
# and bias-variance balance
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
    dtest = xgb.DMatrix(X_test, label=y_test)
    train_auc = []
    test_auc = []

    nrs = np.arange(1, 20) * 20
    param = {'silent': 1, 'max_depth':1, 'eval_metric':'auc', 'eta':0.2} # 
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]
    result = {}
    '''
    lr = [0.3] * 30
    lr.extend([0.2] * 20)
    lr.extend([0.1] * 10)
    '''

    bst = xgb.train(param, dtrain, num_boost_round=nrs[-1], evals=eval_list,
            early_stopping_rounds=30, evals_result=result, verbose_eval=10,
            learning_rates=None)

    plt.plot(result['train']['auc'], '--', lw=1, color=(0.6, 0.6, 0.6), label='train')
    plt.plot(result['eval']['auc'], '-', lw=1, color=(0.0, 0.0, 0.6), label='test')

plt.xlabel('rounds')
plt.ylabel('auc')
plt.legend(loc="lower right")
plt.grid()
plt.show()
