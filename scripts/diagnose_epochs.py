"""Diagnostic: does a second epoch over sparse ids overfit? (the "one-epoch" effect in CTR models)

    python scripts/diagnose_epochs.py --data data/synth.parquet
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convertiq.data import DENSE, default_buckets, to_tensors  # noqa: E402
from convertiq.experiments import split_by_day  # noqa: E402
from convertiq.metrics import nce, summary  # noqa: E402
from convertiq.models import DCNv2  # noqa: E402
from convertiq.train import bce_logits, fit, predict  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/synth.parquet")
a = p.parse_args()
df = pd.read_parquet(a.data)
tr, _, te = split_by_day(df)
b = default_buckets(df)
T, E = to_tensors(tr, b), to_tensors(te, b)
y = E.click.numpy()
base = float(T.click.mean())
print(f"train CTR {base:.4f}, test CTR {y.mean():.4f}, constant-prediction NCE {nce(y, np.full(len(y), base), base):.4f}")
for emb, wd in [(12, 1e-6), (8, 1e-5)]:
    m, opt = DCNv2(b, len(DENSE), emb), None
    for ep in range(3):
        opt = fit(m, T, lambda mm, s, d, c, p: bce_logits(mm(s, d), c), 1, weight_decay=wd, seed=ep, opt=opt, verbose=False)
        r = summary(y, predict(m, E), base)
        print(f"emb={emb} wd={wd} epoch {ep + 1}: AUC {r['auc']}  NCE {r['nce']}  calib {r['calibration_ratio']}", flush=True)
