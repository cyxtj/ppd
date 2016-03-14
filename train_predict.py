import numpy as np
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc

def test(X, y, w):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    skf = StratifiedKFold(y, n_folds=8)

    # from sklearn import svm
    # wclf = svm.SVC(kernel='linear', class_weight={1: 10}) # svm don't provide proba

    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=100)

    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)

    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(loss='exponential')
    # loss='exponential' is bad, auc=0.56


    for i, (train_index, test_index) in enumerate(skf):
        print i
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w_train, w_test = w[train_index], w[test_index]
        clf.fit(X_train, y_train, w_train)
        probas_ = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        break

    mean_tpr /= skf.n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

