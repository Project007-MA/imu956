from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix
import numpy as np

def metric(a, b, c, d, ln, alpha=None, beta=None, cond=False):
    if cond:
        b /= ln ** 1
        c /= ln ** alpha if alpha is not None else 1
        d /= ln ** beta if beta is not None else 1
    else:
        w = 0.94  # w is not used

    # Calculating all metrics as usual
    sensitivity = a / (a + d) if (a + d) > 0 else 0
    specificity = b / (b + c) if (b + c) > 0 else 0
    precision = a / (a + c) if (a + c) > 0 else 0
    recall = sensitivity
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    e_measure = 1 - (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (a + b) / (a + b + c + d) if (a + b + c + d) > 0 else 0
    Rand_index = accuracy ** 0.5
    mcc = ((a * b) - (c * d)) / (((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5) if (((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5) > 0 else 0
    fpr = c / (c + b) if (c + b) > 0 else 0
    fnr = d / (d + a) if (d + a) > 0 else 0
    npv = b / (b + d) if (b + d) > 0 else 0
    fdr = c / (c + a) if (c + a) > 0 else 0

    # Convert all metrics to percentage
    metrics_list = [accuracy * 100, precision * 100, sensitivity * 100, specificity * 100, f_measure * 100,
                    mcc * 100, npv * 100, fpr * 100, fnr * 100]

    return metrics_list


def multi_confu_matrix(Y_test, Y_pred, *args):
    cm = mcm(Y_test, Y_pred)
    TN, FP, FN, TP = 0, 0, 0, 0

    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]

    return metric(TP, TN, FP, FN, len(cm), *args)


def confu_matrix(Y_test, Y_pred, *args):
    cm = confusion_matrix(Y_test, Y_pred)

    if cm.shape == (2, 2):  # Binary classification case
        TN, FP, FN, TP = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        return metric(TP, TN, FP, FN, len(cm), *args)

    else:  # Multi-class classification case
        # Summing TP, TN, FP, FN across all classes
        mcm_result = mcm(Y_test, Y_pred)
        TN, FP, FN, TP = 0, 0, 0, 0

        for i in range(len(mcm_result)):  # Iterating over each class's confusion matrix
            TN += mcm_result[i][0][0]
            FP += mcm_result[i][0][1]
            FN += mcm_result[i][1][0]
            TP += mcm_result[i][1][1]

        return metric(TP, TN, FP, FN, len(mcm_result), *args)


