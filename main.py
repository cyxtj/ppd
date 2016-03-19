# coding=utf8
import numpy as np
import pandas as pd

import h5py 
# fr = h5py.File(r'Data\Train\X.h5', 'r')
fr = h5py.File(r'Data/Train/X.h5', 'r')
X = fr['X'].value
y = fr['y'].value.astype(bool)
w = fr['w'].value



if __name__ == '__main__':
    import sys
    clf_name = sys.argv[1]
    print clf_name + '======================='
    # from sklearn import svm
    # wclf = svm.SVC(kernel='linear', class_weight={1: 10}) # svm don't provide proba

    if ckf_name == 'XGB':
        from train_predict import test_xgb
        test_xgb(X, y, w)

    sample_weighted = True
    if clf_name =='Ada':
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100)

    elif clf_name == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=0)

    elif clf_name == 'GB':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=500)
        # loss='exponential' is bad, auc=0.56

    elif clf_name == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)

    elif clf_name == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(n_jobs=-1, class_weight='balanced', penalty='l1')
        sample_weighted = False

    from train_predict import test
    test(X, y, w, clf, sample_weighted)
