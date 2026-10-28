"""Ali-CCP loader and a synthetic ad-impression generator with the same schema.

Ali-CCP (Alibaba Click and Conversion Prediction) is the standard public dataset
for click + conversion multi-task models. This uses the widely shared
preprocessed CSV layout: `click`, `purchase`, 23 sparse id fields and 8 dense
fields. Ali-CCP has no timestamps, so delayed-feedback and incremental-training
experiments run on the synthetic generator, which adds `ts`, `conv_ts`, new ads
appearing every day and ad quality that drifts over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

SPARSE = ["101", "121", "122", "124", "125", "126", "127", "128", "129", "205", "206", "207", "210",
          "216", "508", "509", "702", "853", "301", "109_14", "110_14", "127_14", "150_14"]
DENSE = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
HOUR = 3600
DAY = 86400


def load_aliccp(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    for c in SPARSE:
        df[c] = df[c].fillna(0).astype(np.int64)
    for c in DENSE:
        df[c] = df[c].fillna(0.0).astype(np.float32)
    df["purchase"] = (df.purchase * df.click).astype(np.int8)  # a purchase without a click is a logging artefact
    return df


@dataclass
class SynthConfig:
    n_users: int = 20_000
    n_ads_initial: int = 2_000
    new_ads_per_day: int = 150
    n_campaigns: int = 300
    n_categories: int = 40
    days: int = 14
    impressions_per_day: int = 60_000
    dim: int = 12
    conv_delay_mean_h: float = 30.0     # exponential delay between click and conversion
    quality_drift: float = 0.15         # daily random-walk step of ad quality
    seed: int = 11


def _sig(x):
    return 1 / (1 + np.exp(-x))


def generate(cfg: SynthConfig = SynthConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    d = cfg.dim
    cat_vec = rng.normal(0, 1, (cfg.n_categories, d))
    user_vec = rng.normal(0, 1, (cfg.n_users, d))
    user_vec /= np.linalg.norm(user_vec, axis=1, keepdims=True)
    user_buy = rng.normal(0, 0.6, cfg.n_users)            # some users convert more
    user_age = rng.integers(1, 8, cfg.n_users)
    user_gender = rng.integers(1, 3, cfg.n_users)
    user_city = rng.integers(1, 60, cfg.n_users)

    n_ads = cfg.n_ads_initial + cfg.new_ads_per_day * cfg.days
    ad_cat = rng.integers(0, cfg.n_categories, n_ads)
    ad_vec = cat_vec[ad_cat] + rng.normal(0, 0.6, (n_ads, d))
    ad_vec /= np.linalg.norm(ad_vec, axis=1, keepdims=True)
    ad_campaign = rng.integers(0, cfg.n_campaigns, n_ads)
    ad_brand = rng.integers(0, 500, n_ads)
    ad_shop = rng.integers(0, 2000, n_ads)
    ad_born = np.concatenate([np.full(cfg.n_ads_initial, -1),
                              np.repeat(np.arange(cfg.days), cfg.new_ads_per_day)])
    ctr_q = rng.normal(0, 0.5, n_ads)
    cvr_q = rng.normal(0, 0.6, n_ads)   # landing-page quality: matters only after the click
    conv_w = rng.normal(0, 1, d)
    conv_w *= 0.8 / np.linalg.norm(conv_w)      # what makes a click turn into a purchase differs from what makes a click

    frames = []
    for day in range(cfg.days):
        ctr_q += rng.normal(0, cfg.quality_drift, n_ads)
        live = np.flatnonzero(ad_born <= day)
        n = cfg.impressions_per_day
        u = rng.integers(0, cfg.n_users, n)
        # Logged policy: the auction prefers ads that look relevant, so shown ads are not random.
        slate = rng.choice(live, (n, 8))
        aff = 2.0 * np.einsum("nd,nkd->nk", user_vec[u], ad_vec[slate]) + 0.5 * ctr_q[slate]
        pick = (np.exp(aff) / np.exp(aff).sum(1, keepdims=True)).cumsum(1) > rng.random((n, 1))
        a = slate[np.arange(n), pick.argmax(1)]

        affinity = (user_vec[u] * ad_vec[a]).sum(1)
        p_click = _sig(2.5 * affinity + ctr_q[a] - 4.0)
        click = rng.random(n) < p_click
        p_cvr = _sig(1.5 * affinity + 2.0 * (user_vec[u] @ conv_w) + cvr_q[a] + user_buy[u] - 2.6)
        purchase = click & (rng.random(n) < p_cvr)
        ts = day * DAY + rng.integers(0, DAY, n)
        delay = rng.exponential(cfg.conv_delay_mean_h * HOUR, n)
        conv_ts = np.where(purchase, ts + delay, np.nan)

        # Historical-rate style dense features, noisy versions of the truth, like real CTR counters.
        f = pd.DataFrame({
            "101": u + 1, "121": user_age[u], "122": user_gender[u], "124": user_city[u],
            "125": (user_buy[u] > 0).astype(int) + 1, "126": rng.integers(1, 4, n), "127": rng.integers(1, 4, n),
            "128": rng.integers(1, 3, n), "129": rng.integers(1, 5, n),
            "205": a + 1, "206": ad_cat[a] + 1, "207": ad_shop[a] + 1, "210": ad_campaign[a] + 1,
            "216": ad_brand[a] + 1, "508": ad_cat[a] * 10 + user_age[u], "509": ad_brand[a] % 97 + 1,
            "702": ad_cat[a] * 3 + user_gender[u], "853": ad_shop[a] % 211 + 1, "301": rng.integers(1, 4, n),
            "109_14": ad_cat[a] + 1, "110_14": ad_brand[a] % 50 + 1, "127_14": ad_shop[a] % 31 + 1,
            "150_14": ad_campaign[a] % 23 + 1,
            "D109_14": np.clip(_sig(2.5 * affinity - 3.5) + rng.normal(0, 0.03, n), 0, 1),
            "D110_14": np.clip(_sig(ctr_q[a] - 2.5) + rng.normal(0, 0.02, n), 0, 1),
            "D127_14": np.clip(_sig(cvr_q[a] - 2) + rng.normal(0, 0.05, n), 0, 1),
            "D150_14": np.clip(rng.normal(0.3, 0.1, n), 0, 1),
            "D508": rng.random(n) * 0.2, "D509": rng.random(n) * 0.2, "D702": rng.random(n) * 0.2,
            "D853": np.clip(user_buy[u] * 0.1 + 0.2 + rng.normal(0, 0.05, n), 0, 1),
            "click": click.astype(np.int8), "purchase": purchase.astype(np.int8),
            "day": day, "ts": ts.astype(np.float64), "conv_ts": conv_ts,
        })
        frames.append(f)
    df = pd.concat(frames, ignore_index=True).sort_values("ts", kind="stable").reset_index(drop=True)
    for c in DENSE:
        df[c] = df[c].astype(np.float32)
    return df


@dataclass
class Tensors:
    sparse: torch.Tensor
    dense: torch.Tensor
    click: torch.Tensor
    purchase: torch.Tensor

    def __len__(self):
        return len(self.click)

    def subset(self, idx) -> "Tensors":
        idx = torch.as_tensor(idx)
        return Tensors(self.sparse[idx], self.dense[idx], self.click[idx], self.purchase[idx])


def to_tensors(df: pd.DataFrame, buckets: list[int], purchase_col: str = "purchase") -> Tensors:
    """Hash every sparse id into its field's bucket range so unseen ids still get a slot."""
    sp = np.stack([df[c].to_numpy(np.int64) % b for c, b in zip(SPARSE, buckets)], 1)
    return Tensors(torch.as_tensor(sp), torch.as_tensor(df[DENSE].to_numpy(np.float32, copy=True)),
                   torch.as_tensor(df.click.to_numpy(np.float32, copy=True)),
                   torch.as_tensor(df[purchase_col].to_numpy(np.float32, copy=True)))


def default_buckets(df: pd.DataFrame, cap: int = 200_000) -> list[int]:
    return [int(min(cap, max(16, df[c].max() * 2 + 1))) for c in SPARSE]
