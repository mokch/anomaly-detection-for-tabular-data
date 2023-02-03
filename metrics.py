from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import numpy as np


def accuracy_score(scores, labels, ratio):
    thresh = np.percentile(scores, ratio*100)
    auroc = roc_auc_score(labels, scores)
    y_pred = (scores >= thresh).astype(int)
    y_true = np.array(labels).astype(int)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f_score, support, auroc