from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def f_score(scores, labels, ratio):
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = labels.astype(int)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return f_score