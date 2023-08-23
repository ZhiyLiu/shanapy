import numpy as np
from dwd.socp_dwd import DWD
from sklearn import metrics
from sklearn.model_selection import train_test_split

class Classification:
    def __init__():
        pass
    @staticmethod
    def partition(X, y, test_size=0.2, rand_seed=0):
        pos_X = X[y==1, :]
        neg_X = X[y==0, :]
        num_pos = pos_X.shape[0]
        num_neg = neg_X.shape[0]
        pos_y = [1] * num_pos
        neg_y = [0] * num_neg
        # train_pos_num = int(num_pos * train_size) + 1
        # train_neg_num = num_neg - (num_pos - train_pos_num)
        X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(\
                                                            pos_X, pos_y, test_size=test_size, random_state=rand_seed)
        neg_test_size = (1-num_pos * (1-test_size) / num_neg)
        X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split(\
                                                            neg_X, neg_y, test_size=test_size, random_state=rand_seed)
        X_train = np.concatenate((X_train_pos, X_train_neg))
        y_train = np.array([1] * X_train_pos.shape[0] + [0] * X_train_neg.shape[0])
        X_test = np.concatenate((X_test_pos, X_test_neg))
        y_test = np.array([1] * X_test_pos.shape[0] + [0] * X_test_neg.shape[0])
        print("Partition of data: Training positive (%d/%d), negative (%d/%d). Test pos (%d/%d), neg (%d/%d)."%(X_train_pos.shape[0], num_pos, X_train_neg.shape[0], num_neg, X_test_pos.shape[0], num_pos, X_test_neg.shape[0], num_neg))
        return X_train, X_test, y_train, y_test

    @staticmethod
    def classify(X, y, test_size = 0.2):
        aucs = []
        for rand_seed in range(1000):
            print(rand_seed)
            X_train, X_test, y_train, y_test = Classification.partition(X, y, test_size, rand_seed)
            dwd = DWD().fit(X_train, y_train)
            y_pred = dwd.decision_function(X_test)
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)

            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
        return np.mean(aucs)
