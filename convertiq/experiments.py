"""The four ConvertIQ experiments. Each returns a JSON-able dict.

All models train for ONE epoch: with sparse-id embeddings a second pass overfits
(scripts/diagnose_epochs.py shows test AUC 0.711 -> 0.688 -> 0.635 over epochs 1-3).

1. multitask     single-task vs shared-bottom vs MMoE, with and without ESMM
2. calibration   negative downsampling breaks calibration; analytic, Platt and isotonic fix it
3. delayed       conversions arrive hours later: naive vs waiting vs fake-negative weighting
4. incremental   ad quality drifts daily: frozen vs warm-start vs full retrain
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .calibration import Isotonic, Platt, segment_calibration, undo_negative_downsampling
from .data import DAY, DENSE, SPARSE, default_buckets, to_tensors
from .metrics import summary
from .models import DCNv2, MultiTask
from .train import bce_logits, esmm_loss, fit, fnw_loss, predict

N_DENSE = len(DENSE)


def _single_loss(target: str):
    def f(m, s, d, c, p):
        return bce_logits(m(s, d), c if target == "click" else p)
    return f


def _esmm(m, s, d, c, p):
    a, b = m(s, d)
    return esmm_loss(a, b, c, p)


def _direct(m, s, d, c, p):
    a, b = m(s, d)
    return bce_logits(a, c) + bce_logits(b, p)


def split_by_day(df: pd.DataFrame, val_days: int = 2, test_days: int = 2):
    if "day" not in df:  # Ali-CCP has no time: a fixed random split
        r = np.random.default_rng(0).random(len(df))
        return df[r < 0.8], df[(r >= 0.8) & (r < 0.9)], df[r >= 0.9]
    last = df.day.max()
    te = df[df.day > last - test_days]
    va = df[(df.day > last - test_days - val_days) & (df.day <= last - test_days)]
    tr = df[df.day <= last - test_days - val_days]
    return tr, va, te


# ----------------------------------------------------------------------------- 1
def multitask(df: pd.DataFrame, epochs: int = 1, seed: int = 0) -> dict:
    tr_df, _, te_df = split_by_day(df)
    buckets = default_buckets(df)
    tr, te = to_tensors(tr_df, buckets), to_tensors(te_df, buckets)
    clicked = te.click.numpy() > 0.5
    y_click, y_buy = te.click.numpy(), te.purchase.numpy()
    results, timings = {}, {}

    def record(name, p_ctr, p_ctcvr, p_cvr):
        results[name] = {
            "ctr": summary(y_click, p_ctr),
            "ctcvr": summary(y_buy, p_ctcvr),
            "cvr_auc_on_clicks": summary(y_buy[clicked], p_cvr[clicked])["auc"],
        }

    print("  single-task DCN v2 (CVR trained on clicks only)", flush=True)
    t0 = time.time()
    ctr_m, cvr_m = DCNv2(buckets, N_DENSE), DCNv2(buckets, N_DENSE)
    fit(ctr_m, tr, _single_loss("click"), epochs, seed=seed)
    fit(cvr_m, tr.subset(np.flatnonzero(tr.click.numpy() > 0.5)), _single_loss("purchase"), epochs, seed=seed)
    p_ctr, p_cvr = predict(ctr_m, te), predict(cvr_m, te)
    timings["single"] = time.time() - t0
    record("single_task_dcn", p_ctr, p_ctr * p_cvr, p_cvr)

    for kind in ("shared", "mmoe"):
        print(f"  {kind} + ESMM", flush=True)
        t0 = time.time()
        m = MultiTask(buckets, N_DENSE, kind)
        fit(m, tr, _esmm, epochs, seed=seed)
        a, b = predict(m, te)
        timings[f"{kind}_esmm"] = time.time() - t0
        record(f"{kind}_esmm", a, a * b, b)
        if kind == "mmoe":
            with torch.no_grad():
                g = m.gate_weights(te.sparse[:20000], te.dense[:20000]).numpy()
            results["mmoe_gate_weights"] = {"ctr_task": g[0].round(3).tolist(), "cvr_task": g[1].round(3).tolist()}

    print("  mmoe, direct CTCVR head (no ESMM)", flush=True)
    m = MultiTask(buckets, N_DENSE, "mmoe")
    fit(m, tr, _direct, epochs, seed=seed)
    a, b = predict(m, te)
    record("mmoe_direct_ctcvr", a, b, b / np.clip(a, 1e-6, None))
    results["train_rows"], results["test_rows"] = len(tr), len(te)
    results["train_seconds"] = {k: round(v, 1) for k, v in timings.items()}
    return results


# ----------------------------------------------------------------------------- 2
def calibration(df: pd.DataFrame, keep_rate: float = 0.1, epochs: int = 1, seed: int = 0) -> dict:
    """CTR model trained on downsampled negatives, then recalibrated on the validation window."""
    tr_df, va_df, te_df = split_by_day(df)
    rng = np.random.default_rng(seed)
    tr_ds = tr_df[(tr_df.click == 1) | (rng.random(len(tr_df)) < keep_rate)]
    buckets = default_buckets(df)
    tr, va, te = (to_tensors(x, buckets) for x in (tr_ds, va_df, te_df))
    m = DCNv2(buckets, N_DENSE)
    fit(m, tr, _single_loss("click"), epochs, seed=seed)
    p_va, p_te = predict(m, va), predict(m, te)
    y_va, y_te = va.click.numpy(), te.click.numpy()

    variants = {
        "raw (trained on 10% of negatives)": p_te,
        "analytic downsampling correction": undo_negative_downsampling(p_te, keep_rate),
        "platt (fit on val window)": Platt().fit(p_va, y_va)(p_te),
        "isotonic (fit on val window)": Isotonic().fit(p_va, y_va)(p_te),
    }
    seg_col = "301"
    out = {name: summary(y_te, p) for name, p in variants.items()}
    out["per_segment_calibration_ratio"] = {
        name: segment_calibration(y_te, p, te_df[seg_col].to_numpy()) for name, p in variants.items()}
    out["kept_training_rows"] = int(len(tr_ds))
    out["full_training_rows"] = int(len(tr_df))
    return out


# ----------------------------------------------------------------------------- 3
def delayed_feedback(df: pd.DataFrame, cutoff_day: int | None = None, epochs: int = 1, seed: int = 0) -> dict:
    """Train at time T on the labels known by T; score the next day against final labels."""
    if "conv_ts" not in df:
        return {"skipped": "needs timestamps; run on the synthetic data"}
    cutoff_day = cutoff_day if cutoff_day is not None else int(df.day.max()) - 2
    T = (cutoff_day + 1) * DAY
    train = df[df.ts < T].copy()
    test = df[df.day == cutoff_day + 1]
    buckets = default_buckets(df)
    observed = (train.purchase == 1) & (train.conv_ts <= T)
    train["obs"] = observed.astype(np.int8)
    te = to_tensors(test, buckets)
    y = te.purchase.numpy()

    def run(frame, label_col, loss):
        t = to_tensors(frame, buckets, purchase_col=label_col)
        m = DCNv2(buckets, N_DENSE)
        fit(m, t, loss, epochs, seed=seed, verbose=False)
        return summary(y, predict(m, te))

    def target(lossf):
        return lambda m, s, d, c, p: lossf(m(s, d), p)

    out = {"missing_label_share_at_T": round(1 - float(observed.sum()) / max(int(train.purchase.sum()), 1), 4)}
    print("  oracle / naive / wait windows / FNW", flush=True)
    out["oracle_final_labels"] = run(train, "purchase", target(bce_logits))
    out["naive_observed_labels"] = run(train, "obs", target(bce_logits))
    for hours in (24, 72):
        out[f"wait_{hours}h"] = run(train[train.ts < T - hours * 3600], "obs", target(bce_logits))
    # FNW stream: every impression once as a negative, observed conversions re-inserted as positives.
    neg = train.assign(obs=0)
    pos = train[train.obs == 1]
    stream = pd.concat([neg, pos], ignore_index=True)
    out["fake_negative_weighted"] = run(stream, "obs", target(fnw_loss))
    return out


# ----------------------------------------------------------------------------- 4
def incremental(df: pd.DataFrame, warm_days: int = 7, seed: int = 0) -> dict:
    """Ad quality drifts every day and new ads launch daily. How stale does a frozen model get?"""
    if "day" not in df:
        return {"skipped": "needs day column; run on the synthetic data"}
    buckets = default_buckets(df)
    days = sorted(df.day.unique())
    by_day = {d: to_tensors(df[df.day == d], buckets) for d in days}
    cat = lambda ts: type(ts[0])(*(torch.cat(z) for z in zip(*[(t.sparse, t.dense, t.click, t.purchase) for t in ts])))  # noqa: E731

    init = cat([by_day[d] for d in days[:warm_days]])
    frozen = DCNv2(buckets, N_DENSE)
    fit(frozen, init, _single_loss("click"), 1, seed=seed, verbose=False)
    warm = DCNv2(buckets, N_DENSE)
    warm.load_state_dict(frozen.state_dict())
    opt = torch.optim.Adam(warm.parameters(), lr=1e-3)

    rows, secs = [], {"warm_start": 0.0, "full_retrain": 0.0}
    for d in days[warm_days:]:
        te = by_day[d]
        y = te.click.numpy()
        row = {"day": int(d), "frozen": summary(y, predict(frozen, te))["nce"],
               "warm_start": summary(y, predict(warm, te))["nce"]}
        t0 = time.time()
        full = DCNv2(buckets, N_DENSE)
        fit(full, cat([by_day[x] for x in days if x < d]), _single_loss("click"), 1, seed=seed, verbose=False)
        secs["full_retrain"] += time.time() - t0
        row["full_retrain"] = summary(y, predict(full, te))["nce"]
        rows.append(row)
        print(f"  day {d}: {row}", flush=True)
        # After scoring day d, today's labels arrive: one warm-start pass on them.
        t0 = time.time()
        fit(warm, by_day[d], _single_loss("click"), 1, seed=seed + int(d), opt=opt, verbose=False)
        secs["warm_start"] += time.time() - t0
    mean = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("frozen", "warm_start", "full_retrain")}
    return {"per_day_nce": rows, "mean_nce": mean, "train_seconds": {k: round(v, 1) for k, v in secs.items()},
            "note": "full_retrain is retrained from scratch on all prior days before each scored day"}
