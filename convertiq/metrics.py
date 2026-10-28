"""Offline metrics for calibrated probability models.

NCE (normalized cross-entropy, He et al. 2014) is the standard offline metric
for ad click and conversion models: log loss divided by the entropy of the base rate. Here the base rate
is the evaluation window's own, so 1.0 is the best constant predictor even when CTR
drifts between training and test. Below 1.0 means
the model beats always predicting the average; lower is better. AUC only checks
ordering, so calibration (predicted / actual conversions) and ECE sit beside it.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def logloss(y, p) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nce(y, p, base_rate: float | None = None) -> float:
    b = float(np.mean(y)) if base_rate is None else base_rate
    b = min(max(b, 1e-7), 1 - 1e-7)
    h = -(b * np.log(b) + (1 - b) * np.log(1 - b))
    return logloss(y, p) / h


def calibration_ratio(y, p) -> float:
    return float(np.sum(p) / max(np.sum(y), 1e-9))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error over equal-mass bins."""
    order = np.argsort(p)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total / len(p))


def summary(y, p, base_rate=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
            "nce": round(nce(y, p, base_rate), 4),
            "logloss": round(logloss(y, p), 5),
            "calibration_ratio": round(calibration_ratio(y, p), 4),
            "ece": round(ece(y, p), 5),
            "positives": int(y.sum()), "rows": int(len(y))}
