from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from convertiq.calibration import Isotonic, Platt, undo_negative_downsampling
from convertiq.data import DENSE, SPARSE, SynthConfig, default_buckets, generate, load_aliccp, to_tensors
from convertiq.metrics import calibration_ratio, ece, nce
from convertiq.models import DCNv2, MultiTask
from convertiq.train import esmm_loss, fit, fnw_loss, predict


@pytest.fixture(scope="module")
def df():
    return generate(SynthConfig(n_users=800, n_ads_initial=200, new_ads_per_day=20, days=4, impressions_per_day=6000))


def test_schema_matches_aliccp(df, tmp_path):
    assert all(c in df for c in SPARSE + DENSE + ["click", "purchase"])
    assert (df.purchase <= df.click).all()  # no purchase without a click
    p = tmp_path / "a.csv"
    df[["click", "purchase"] + SPARSE + DENSE].to_csv(p, index=False)
    back = load_aliccp(p)
    assert back.shape[0] == len(df)


def test_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    assert nce(y, np.full(4, 0.5)) == pytest.approx(1.0)        # predicting the base rate scores exactly 1
    assert nce(y, np.array([.1, .1, .9, .9])) < 1.0
    assert calibration_ratio(y, np.array([.5, .5, .5, .5])) == pytest.approx(1.0)
    assert ece(y.astype(float), y.astype(float)) == 0.0


def test_downsampling_correction_restores_rate():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.01, 0.1, 100_000)
    keep = 0.1
    p_ds = p_true / (p_true + (1 - p_true) * keep)  # what a model trained on 10% of negatives learns
    np.testing.assert_allclose(undo_negative_downsampling(p_ds, keep), p_true, rtol=1e-9)


def test_platt_and_isotonic_fix_scaled_scores():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.3, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    biased = np.clip(p * 2.5, 0, 0.99)
    assert calibration_ratio(y, biased) > 2
    for cal in (Platt(), Isotonic()):
        assert abs(calibration_ratio(y, cal.fit(biased, y)(biased)) - 1) < 0.05


def test_fnw_weights():
    logit = torch.tensor([0.0, 0.0])
    y = torch.tensor([1.0, 0.0])
    # p = 0.5: positive weight 1.5, negative weight 0.75
    expect = (1.5 * np.log(2) + 0.75 * np.log(2)) / 2
    assert float(fnw_loss(logit, y)) == pytest.approx(expect, rel=1e-5)


def test_esmm_ctcvr_is_product():
    a, b = torch.tensor([0.0]), torch.tensor([0.0])
    loss = esmm_loss(a, b, torch.tensor([1.0]), torch.tensor([1.0]))
    assert float(loss) == pytest.approx(np.log(2) + np.log(4), rel=1e-5)


def test_models_train_and_predict(df):
    b = default_buckets(df)
    t = to_tensors(df, b)
    m = MultiTask(b, len(DENSE), "mmoe")
    fit(m, t, lambda mm, s, d, c, p: esmm_loss(*mm(s, d), c, p), epochs=1, batch=2048, verbose=False)
    a, c = predict(m, t)
    assert a.shape == c.shape == (len(t),) and np.all((a > 0) & (a < 1))
    s = DCNv2(b, len(DENSE))
    fit(s, t, lambda mm, sp, d, c, p: torch.nn.functional.binary_cross_entropy_with_logits(mm(sp, d), c),
        epochs=2, batch=2048, verbose=False)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(t.click.numpy(), predict(s, t)) > 0.6
