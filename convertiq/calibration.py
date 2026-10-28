"""Post-hoc calibration layers fitted on a held-out window, applied to the next one.

Standard practice in ad ranking: a small model (Platt scaling, isotonic
regression or a tiny network) on top of the ranker so predicted conversions
match realised conversions, treated as a constraint in A/B tests.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def __call__(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-7, y_max=1 - 1e-7).fit(p, y)
        return self

    def __call__(self, p):
        return self.iso.predict(p)


def undo_negative_downsampling(p, keep_rate: float):
    """Analytic correction when negatives were kept with probability `keep_rate` at training time."""
    return p / (p + (1 - p) / keep_rate)


def segment_calibration(y, p, segments) -> dict:
    """Predicted / actual positives per segment: the check that must hold for every business segment."""
    out = {}
    for s in np.unique(segments):
        m = segments == s
        if y[m].sum() >= 20:
            out[str(s)] = round(float(p[m].sum() / y[m].sum()), 3)
    return out
