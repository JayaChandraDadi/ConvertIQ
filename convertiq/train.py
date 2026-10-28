"""Training loops and losses."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import Tensors


def bce_logits(logit, y, weight=None):
    return F.binary_cross_entropy_with_logits(logit, y, weight=weight)


def esmm_loss(ctr_logit, cvr_logit, click, purchase):
    """pCTR on all impressions + pCTCVR = pCTR * pCVR on all impressions."""
    p_ctr = torch.sigmoid(ctr_logit)
    p_ctcvr = (p_ctr * torch.sigmoid(cvr_logit)).clamp(1e-7, 1 - 1e-7)
    return bce_logits(ctr_logit, click) + F.binary_cross_entropy(p_ctcvr, purchase)


def fnw_loss(logit, y):
    """Fake negative weighted loss (Ktena et al. 2019) for delayed conversions.

    Every impression first enters the stream as a negative; a later conversion
    re-inserts it as a positive. Under that biased stream, importance weights of
    (1 + p) for positives and (1 - p)(1 + p) for negatives recover an unbiased
    estimate of the true loss. p is the model's own stop-gradient prediction.
    """
    p = torch.sigmoid(logit).detach()
    w = torch.where(y > 0.5, 1 + p, (1 - p) * (1 + p))
    return bce_logits(logit, y, weight=w)


def batches(n: int, batch: int, shuffle: bool, seed: int):
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    for s in range(0, n, batch):
        yield torch.as_tensor(idx[s: s + batch])


def fit(model, data: Tensors, loss_fn, epochs: int = 2, batch: int = 4096, lr: float = 2e-3,
        weight_decay: float = 1e-6, seed: int = 0, opt=None, verbose: bool = True):
    """`loss_fn(model, sparse, dense, click, purchase) -> loss`. Returns the optimizer for warm starts."""
    torch.manual_seed(seed)
    opt = opt or torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b in batches(len(data), batch, True, seed + ep):
            loss = loss_fn(model, data.sparse[b], data.dense[b], data.click[b], data.purchase[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
            n += len(b)
        if verbose:
            print(f"    epoch {ep + 1}: loss {tot / n:.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return opt


@torch.no_grad()
def predict(model, data: Tensors, batch: int = 65536):
    """Returns sigmoid outputs: one array for single-task models, a tuple for multi-task."""
    model.eval()
    outs = []
    for b in batches(len(data), batch, False, 0):
        o = model(data.sparse[b], data.dense[b])
        outs.append(tuple(torch.sigmoid(x).numpy() for x in o) if isinstance(o, tuple) else torch.sigmoid(o).numpy())
    if isinstance(outs[0], tuple):
        return tuple(np.concatenate(z) for z in zip(*outs))
    return np.concatenate(outs)
