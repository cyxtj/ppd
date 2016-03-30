import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import xgboost as xgb

def test(X, y, w, clf, sample_weighted=True):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    skf = StratifiedKFold(y, n_folds=8)

    for i, (train_index, test_index) in enumerate(skf):
        print i
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w_train, w_test = w[train_index], w[test_index]
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if sample_weighted:
            clf.fit(X_train, y_train, w_train)
        else:
            clf.fit(X_train, y_train)
        probas_ = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        print roc_auc
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= skf.n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example'+str(clf))
    plt.legend(loc="lower right")
    plt.figtext(0, .5, str(clf))
    plt.show()

def test_xgb(X, y, w):
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
        # dtrain = xgb.DMatrix(X_train, label=y_train) #, weight=w_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # param = {'silent': 1,  'eval_metric':'auc',#  'nthread':30,
        #         'nround':600, 'early_stopping_rounds':30,
        #         'colsample_bytree': 0.8615941212305067, 'min_child_weight': 5, 'subsample': 0.5618441782807552, 'eta': 0.08415830935283643, 'max_depth': 2, 'gamma': 0.5258845694703286}
        param = {'silent': 1,  'eval_metric':'auc',#  'nthread':30,
                'max_depth':1, 'eta':0.03, 'nround': 2000, 
                'subsample': 0.9, 'colsample_bytree':0.8, 
                #'min_child_weight': 5
                }

        
        result = {}
        eval_list = [(dtrain, 'train'), (dtest, 'eval')]
        bst = xgb.train(param, dtrain, num_boost_round=param['nround'], evals=eval_list,
                evals_result=result, verbose_eval=50,
                learning_rates=None, early_stopping_rounds=100)

        p = bst.predict(dtest)
        fpr, tpr, thresholds = roc_curve(y_test, p)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.3f)' % (i, roc_auc))
        print 'test auc: %0.3f'%roc_auc, 
        p2 = bst.predict(dtrain)
        fpr, tpr, thresholds = roc_curve(y_train, p2)
        roc_auc = auc(fpr, tpr)
        print 'train auc: %0.3f'%roc_auc
        plt.plot(fpr, tpr, '--', lw=1)
        # test auc vs train auc at each round
        plt.subplot(1, 2, 2)
        plt.plot(result['train']['auc'], '--', lw=1, color=(0.6, 0.6, 0.6))
        plt.plot(result['eval']['auc'], '-', lw=1, color=(0.0, 0.0, 0.6),
                label='%d'%i)
        plt.draw()

    plt.subplot(1, 2, 1)
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
    plt.title(str(param))

    plt.subplot(1, 2, 2)
    plt.legend(loc="lower right")
    plt.grid()

    plt.draw()
    plt.show()
