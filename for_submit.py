# coding=utf8
import numpy as np
import pandas as pd
from scipy import interp
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

# load data
# X1 = pd.read_csv(r'Data/Train/PPD_Training_Master_GBK_3_1_Training_Set.csv', encoding='gbk')
X2 = pd.read_csv(r'Data/Test/PPD_Master_GBK_2_Test_Set.csv')# , encoding='gbk')
from util import load_train, load_test
X_train, y_train, w_train = load_train()
X_test = load_test()
print 'data loaded, transforming...'

''' 3.19 commented
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# train and predict
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(n_jobs=-1, class_weight='balanced', penalty='l1')
clf.fit(X_train, y_train)
probas_ = clf.predict_proba(X_test)

# visualization on training set
fpr, tpr, thresholds = roc_curve(y_train, p2[:, 1])
mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)
print roc_auc
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example'+str(clf))
plt.legend(loc="lower right")
plt.figtext(0, .5, str(clf))
plt.show()

# save result
X2['score'] = probas_[:, 1]
result = X2[['Idx', 'score']]
result['Idx'].astype(int)
result.to_csv('result.csv', float_format='%.4f')
'''

# 3.19 submit
# 3.20 submit change data with Text-Update-PeriodDiff
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
dtest = xgb.DMatrix(X_test)
param = {'silent': 1, 'max_depth':1, 'eta':0.05, 'subsample':0.8,
        'colsample_bytree': 0.8}
bst = xgb.train(param, dtrain, num_boost_round=1000)
p = bst.predict(dtest)
X2['score'] = p
result = X2[['Idx', 'score']]
result['Idx'].astype(int)
result.to_csv('result.csv', float_format='%.4f')
