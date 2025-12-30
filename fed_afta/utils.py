# fed_afta/utils.py
import numpy as np
from sklearn.metrics import accuracy_score

def uncertainty_sampling_proba(probs, k):
    """
    Choose k indices with smallest margin (most uncertain).
    probs: array-like of probabilities for class 1 (shape (n,))
    """
    probs = np.asarray(probs)
    if probs.ndim == 1:
        margins = np.abs(probs - 0.5)
        idx = np.argsort(margins)[:k]
        return idx
    else:
        # multiclass fallback: choose smallest top-2 margin
        top2 = np.partition(-probs, 1, axis=1)[:, :2] * -1
        margins = np.abs(top2[:,0] - top2[:,1])
        idx = np.argsort(margins)[:k]
        return idx

def evaluate_preds(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))