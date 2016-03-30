import numpy as np
import pandas as pd
import json

from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys
import xgboost as xgb

def write_submission(preds, output):
    sample = pd.read_csv('../data/sampleSubmission.csv')
    preds = pd.DataFrame(
        preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(output, index_label='id')


def score(params):
    print '--------------------------------------------------'
    print "Training with params : "
    print params
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])
    params['eval_metric'] = 'auc'
    params['silent'] = 1
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    result = {}
    eval_list = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=1000,
            evals=eval_list, verbose_eval=100, evals_result=result, 
            early_stopping_rounds=50)
    s = model.best_score
    f = open('paramTuneResults.json', 'a')
    result['best_score'] = s
    result['best_iteration'] = model.best_iteration
    result['best_ntree_limit'] = model.best_ntree_limit
    f.write(json.dumps(dict(params, **result)))
    f.write('\n')
    f.close()
    print '## score: ', s, ' ##'
    return {'loss': 1-s, 'status': STATUS_OK, 'result': result}


def optimize(trials):
    space = {
             'eta' : hp.uniform('eta', 0.05, 0.3),
             'max_depth' : hp.quniform('max_depth', 1, 8, int(1)),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.uniform('subsample', 0.5, 1), 
             'gamma' : hp.uniform('gamma', 0.5, 1), 
             'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1), 
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=500)
    print '-------------------------------'
    print 'best parameters are: '
    print best
    return best

from util import load_train
X, y, w = load_train()
print "Splitting data into train and valid ...\n\n"
skf = StratifiedKFold(y, n_folds=8)
for i, (train_index, test_index) in enumerate(skf):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    w_train, w_test = w[train_index], w[test_index]
    break

#Trials object where the history of search will be stored
trials = Trials()

best = optimize(trials)
