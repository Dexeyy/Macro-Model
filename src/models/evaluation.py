from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def evaluate_classifier(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int).reindex(y_true.index).fillna(method="ffill").fillna(method="bfill")
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int).reindex(y_true.index).fillna(method="ffill").fillna(method="bfill")
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm)


def regime_durations(labels: pd.Series) -> Dict[str, float]:
    lab = pd.Series(labels).values
    if len(lab) == 0:
        return {"mean": 0.0, "median": 0.0}
    runs = []
    cur = lab[0]
    k = 1
    for i in range(1, len(lab)):
        if lab[i] == cur:
            k += 1
        else:
            runs.append(k)
            cur = lab[i]
            k = 1
    runs.append(k)
    return {"mean": float(np.mean(runs)), "median": float(np.median(runs))}


