# ConvertIQ

Multi-task click and conversion prediction for ad ranking, with the production problems that come with it: calibration, delayed conversions and daily model drift. PyTorch, on the Ali-CCP schema (Alibaba Click and Conversion Prediction), plus a synthetic generator that adds the timestamps, conversion delays and drifting ad quality that Ali-CCP lacks.

Ad auctions use predicted probabilities directly (bid × pCVR), so the model has to rank well **and** be calibrated. The four experiments here each isolate one part of that.

## Models

All share one input layer (hashed field embeddings, so unseen ad ids still get a slot, plus batch-normalised dense features) and a low-rank **DCN v2** cross network.

| Variant | How the tasks share | Conversion head |
| --- | --- | --- |
| `single_task_dcn` | separate models | CVR trained on clicked rows only |
| `shared_esmm` | shared-bottom MLP, two towers | ESMM: pCTCVR = pCTR × pCVR, trained on all impressions |
| `mmoe_esmm` | 4 experts, a softmax gate per task | ESMM |
| `mmoe_direct_ctcvr` | MMoE | predicts pCTCVR directly |

ESMM (Ma et al. 2018) removes the sample-selection bias of training CVR on clicks while serving it on every impression.

Metrics: **NCE** (normalised cross-entropy: log loss over the entropy of the window's base rate; below 1.0 beats the best constant prediction), AUC, calibration ratio (predicted / actual positives) and ECE.

## Results (synthetic data: 840K impressions, 14 days, CTR 4.3%, CVR given click 14.2%)

Trained on days 1–10, tested on days 13–14. Numbers are from `results/*.json`.

### 1. Multi-task: what sharing buys

| Model | CTR AUC | CTR NCE | CTCVR AUC | CTCVR NCE | CTCVR calibration |
| --- | --- | --- | --- | --- | --- |
| Single-task DCN v2 ×2 | 0.711 | 0.928 | 0.717 | 1.079 | **2.81** |
| Shared-bottom + ESMM | 0.712 | 0.927 | 0.737 | 0.939 | 1.10 |
| MMoE + ESMM | 0.713 | 0.927 | 0.738 | 0.937 | 1.07 |
| **MMoE, direct CTCVR head** | **0.713** | **0.927** | **0.759** | **0.923** | 0.93 |

The single-task pipeline shows sample-selection bias in one number: a CVR model trained only on clicks, served on every impression, overpredicts conversions 2.8x and loses to a constant predictor (NCE above 1). Training both tasks on all impressions fixes it. On this data a direct CTCVR head beat the ESMM factorisation on ranking; ESMM's win is removing the bias, not adding AUC. The MMoE gates did specialise: the conversion task leans on expert 1, the click task on expert 4.

### 2. Calibration: downsampling breaks it, a small layer fixes it

CTR model trained on all clicks and 10% of non-clicks (83K of 600K rows), the usual trick to cut training cost:

| Correction | AUC | NCE | Predicted / actual clicks | ECE |
| --- | --- | --- | --- | --- |
| None | 0.700 | 2.631 | 8.11 | 0.334 |
| Analytic downsampling formula | 0.700 | 0.982 | 1.24 | 0.024 |
| Platt scaling (fit on validation days) | 0.700 | 0.936 | **1.00** | 0.0024 |
| Isotonic regression (fit on validation days) | 0.700 | 0.938 | **1.00** | **0.0015** |

The analytic formula assumes a converged model on a stationary distribution, and leaves 24% overprediction. A calibration layer fit on the most recent window absorbs that plus the CTR drift between training and test, and holds within 2% in every segment of field 301.

### 3. Delayed conversions

Conversions arrive a mean 30 hours after the click, so at training time 11% of the window's conversions are not reported yet:

| Training labels | AUC | NCE | Calibration |
| --- | --- | --- | --- |
| Oracle (all conversions known) | 0.758 | 0.923 | 1.07 |
| Naive (labels known at training time) | 0.757 | 0.925 | **0.90** |
| Wait 24 h, then train | 0.759 | 0.924 | 0.90 |
| Wait 72 h, then train | 0.749 | 0.934 | 0.90 |
| **Fake-negative weighted loss** | 0.756 | 0.924 | **1.05** |

The naive model underpredicts conversions by about 10%, which means underbidding in the auction. Waiting longer does not fix it: the newest data still has missing labels, and waiting 72 hours costs freshness (AUC drops a point). Fake-negative weighting (Ktena et al. 2019) restores calibration to oracle level without waiting. The eval day has 375 conversions, so treat AUC differences under 0.01 as noise.

### 4. Daily drift: warm-start updates

Ad quality drifts daily and 150 new ads launch every day. Mean next-day CTR NCE over days 8–14:

| Update policy | Mean NCE | Training time |
| --- | --- | --- |
| Frozen after day 7 | 0.938 | 0 |
| Full retrain from scratch every day | 0.934 | 75.7 s |
| **Warm start: one pass over yesterday** | **0.933** | **7.7 s** |

Warm-starting from yesterday's checkpoint matches a full daily retrain at a tenth of the cost, and the gap to the frozen model widens over time (0.924 vs 0.931 by day 14).

### Finding: one epoch

`scripts/diagnose_epochs.py` shows the "one-epoch" overfitting known in CTR models with sparse id embeddings: test AUC 0.711 after one epoch, 0.688 after two, 0.635 after three. Every model here trains for one epoch.


## Run it

```bash
pip install -r requirements.txt

python -m convertiq.cli synth --out data/synth.parquet        # 840K impressions, 14 days
python -m convertiq.cli run --data data/synth.parquet --out results

# Ali-CCP (preprocessed CSV with click, purchase, 23 sparse and 8 dense columns):
python -m convertiq.cli run --aliccp data/ali_ccp_train.csv --nrows 3000000 --only multitask calibration

python scripts/diagnose_epochs.py --data data/synth.parquet   # the one-epoch overfitting check
pytest -q
```

Ali-CCP has no timestamps, so `delayed` and `incremental` run on the synthetic data only. Every experiment fits on an M-series MacBook; the full synthetic run takes about 10 minutes on 2 CPU cores.

## Layout

```
convertiq/data.py         Ali-CCP loader, synthetic generator (delays, drift, new ads), hashed tensors
convertiq/models.py       inputs, low-rank DCN v2, single-task, shared-bottom, MMoE
convertiq/train.py        training loop, ESMM loss, fake-negative-weighted loss
convertiq/calibration.py  Platt, isotonic, analytic downsampling correction, per-segment check
convertiq/metrics.py      NCE, AUC, calibration ratio, ECE
convertiq/experiments.py  the four experiments
scripts/diagnose_epochs.py
results/                  JSON from the synthetic run
tests/                    metrics, losses, calibrators, schema, model training
```
